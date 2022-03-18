#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_add(const MatrixCSR<A> *_matrix,
                         const SparseVector<X> *_x,
                         DenseVector<Y> *_y,
                         BinaryOpTAccum _accum,
                         SemiringT _op,
                         Descriptor *_desc,
                         Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();
    auto identity_val = _op.identity();
    auto mul_op = extractMul(_op);

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < y_size; row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
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
    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmspv and BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_or(const MatrixCSR<A> *_matrix,
                        const SparseVector <X> *_x,
                        DenseVector<Y> *_y,
                        BinaryOpTAccum _accum,
                        SemiringT _op,
                        Descriptor *_desc,
                        Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < y_size; row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
            ENT row_start   = _matrix->row_ptr[ind];
            ENT row_end     = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                y_vals[dest_ind] = 1;
            }
        }
    }
    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmspv or BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_critical(const MatrixCSR<A> *_matrix,
                              const SparseVector <X> *_x,
                              DenseVector<Y> *_y,
                              BinaryOpTAccum _accum,
                              SemiringT _op,
                              Descriptor *_desc,
                              Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < y_size; row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
            ENT row_start = _matrix->row_ptr[ind];
            ENT row_end   = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                #pragma omp critical
                {
                    y_vals[dest_ind] = add_op(y_vals[dest_ind], mul_op(x_val, dest_val));
                }
            }
        }
    }
    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmspv critical BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_add_opt(const MatrixCSR<A> *_matrix,
                             const SparseVector <X> *_x,
                             DenseVector<Y> *_y,
                             BinaryOpTAccum _accum,
                             SemiringT _op,
                             Descriptor *_desc,
                             Workspace *_workspace)
{
    /*const X *x_vals = _x->get_vals();
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
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
