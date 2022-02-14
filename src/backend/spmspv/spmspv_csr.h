#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_add(const MatrixCSR<A> *_matrix,
                         const DenseVector<X> *_x,
                         DenseVector<Y> *_y,
                         BinaryOpTAccum _accum,
                         SemiringT _op,
                         Descriptor *_desc,
                         Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    /*cout << _x->get_nvals() << " vs " << _x->get_size() << endl;

    VNT real_nvals = 0;
    for(VNT i = 0; i < _x->get_size(); i++)
        if(x_vals[i] != 0)
            real_nvals++;

    for(VNT i = 0; i < _x->get_size(); i++)
        if(i < 20)
            cout << x_vals[i] << " ";
    cout << endl;
    cout << "real nvals: " << real_nvals << endl;*/

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < _y->get_size(); row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT row = 0; row < _y->get_size(); row++)
        {
            VNT ind = row;
            X x_val = x_vals[row];
            ENT row_start   = _matrix->row_ptr[ind];
            ENT row_end     = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                #pragma omp atomic
                y_vals[dest_ind] += mul_op(x_val, dest_val);
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << "spmspv BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_add_opt(const MatrixCSR<A> *_matrix,
                         const DenseVector<X> *_x,
                         DenseVector<Y> *_y,
                         BinaryOpTAccum _accum,
                         SemiringT _op,
                         Descriptor *_desc,
                         Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_size = _x->get_size();
    VNT y_size = _y->get_size();

    Y *copy;
    MemoryAPI::allocate_array(&copy, y_size*12);

    #pragma omp parallel
    {
        int group_id = omp_get_thread_num() / 4;

        Y *loc_data = &copy[group_id*y_size];
        if((omp_get_thread_num() % 4) == 0)
        {
            for(size_t i = 0; i < y_size; i++)
            {
                loc_data[i] = identity_val;
            }
        }
    }

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int group_id = omp_get_thread_num() / 4;
        X* loc_data = &copy[group_id*x_size];

        #pragma omp for
        for (VNT row = 0; row < y_size; row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT x_pos = 0; x_pos < x_size; x_pos++)
        {
            VNT ind = x_pos;
            X x_val = x_vals[x_pos];
            ENT row_start   = _matrix->row_ptr[ind];
            ENT row_end     = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                #pragma omp atomic
                loc_data[dest_ind] += mul_op(x_val, dest_val);
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << "spmspv BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    for(int group_id = 0; group_id < 12; group_id++)
    {
        Y *loc_data = &copy[group_id*y_size];
        #pragma omp parallel for
        for(size_t i = 0; i < y_size; i++)
        {
            y_vals[i] += loc_data[i];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV(const Matrix<A> *_matrix,
            const DenseVector <X> *_x,
            DenseVector <Y> *_y,
            Descriptor *_desc,
            BinaryOpTAccum _accum,
            SemiringT _op,
            const Vector <M> *_mask)
{
    cout << "doing SpMSpV" << endl;
    if(_mask == NULL) // all active case
    {
        auto add_op = extractAdd(_op);
        /*!
          * /brief atomicAdd() 3+5  = 8
          *        atomicSub() 3-5  =-2
          *        atomicMin() 3,5  = 3
          *        atomicMax() 3,5  = 5
          *        atomicOr()  3||5 = 1
          *        atomicXor() 3^^5 = 0
        */
        int functor = add_op(3, 5);
        if (functor == 8)
        {
            spmspv_unmasked_add(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
            spmspv_unmasked_add_opt(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
        }
        else
        {
            throw "Error in SpMSpV : unsupported additive operation in semiring";
        }
    }
    else
    {
        throw "Error in SpMSpV : Masked SpMSpV not supported yet";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
