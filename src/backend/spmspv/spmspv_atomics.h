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

    Y *tmp_result = (Y*)_workspace->get_shared_one();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    std::set<Y> changed_results;

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
            ENT row_start = _matrix->row_ptr[ind]; // this is actaully col ptr for mxv operation
            ENT row_end   = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
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
void spmspv_unmasked_map(const MatrixCSR<A> *_matrix,
                         const SparseVector <X> *_x,
                         SparseVector<Y> *_y,
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

    Y *tmp_result = (Y*)_workspace->get_shared_one();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    std::unordered_map<VNT, Y> map_output;

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
            ENT row_start = _matrix->row_ptr[ind]; // this is actaully col ptr for mxv operation
            ENT row_end   = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
                A dest_val = _matrix->vals[j];

                #pragma omp critical
                {
                    Y old_val = identity_val;
                    auto find_res = map_output.find(dest_ind);
                    if(find_res != map_output.end())
                    {
                        old_val = (*find_res).second;
                    }
                    map_output[dest_ind] = add_op(old_val, mul_op(x_val, dest_val));
                }
            }
        }
    }

    std::cout << "going to print generated map of size " << map_output.size() << endl;
    _y->clear();
    for(auto [id, val]: map_output)
    {
        //std::cout << id << " - " << val << std::endl;
        _y->push_back(id, val);
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmspv critical map BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
