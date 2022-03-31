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
                A mat_val = _matrix->vals[j];

                #pragma omp atomic
                y_vals[dest_ind] += mul_op(mat_val, x_val);
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

    /*
    vector<ENT> degrees(x_nvals, 0);
    vector<ENT> work_ptrs(x_nvals + 1, 0);

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            degrees[i] = _matrix->row_ptr[ind + 1] - _matrix->row_ptr[ind];
        }
    };

    ParallelPrimitives::exclusive_scan(&degrees[0], &work_ptrs[0], x_nvals);

    int threads_count = omp_get_max_threads();
    ENT approx_nnz_per_thread = (work_ptrs[x_nvals] - 1) / threads_count + 1;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ENT expected_tid_left_border = approx_nnz_per_thread * tid;
        ENT expected_tid_right_border = approx_nnz_per_thread * (tid + 1);

        auto low_pos = std::lower_bound(work_ptrs.begin(), work_ptrs.end(), expected_tid_left_border);
        auto up_pos = std::lower_bound(work_ptrs.begin(), work_ptrs.end(), expected_tid_right_border);

        VNT low_val = min(x_nvals, (VNT)(low_pos - work_ptrs.begin()));
        VNT up_val = min(x_nvals, (VNT)(up_pos - work_ptrs.begin()));

        for (VNT i = low_val; i < up_val; i++)
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
    t2 = omp_get_wtime();
    cout << "LOAD BALANCED SPMSPV BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl << endl;
     */
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
                A mat_val = _matrix->vals[j];

                #pragma omp critical
                {
                    y_vals[dest_ind] = add_op(y_vals[dest_ind], mul_op(mat_val, x_val));
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

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    //#ifdef __USE_TBB__
    //tbb::concurrent_unordered_map<VNT, Y> map_output;
    //#else
    std::unordered_map<VNT, Y> map_output;
    //#endif

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
                A mat_val = _matrix->vals[j];

                #pragma omp critical
                {
                    if(map_output.find(dest_ind) == map_output.end())
                    {
                        map_output[dest_ind] = add_op(identity_val, mul_op(mat_val, x_val));
                    }
                    else
                    {
                        map_output[dest_ind] = add_op(map_output[dest_ind], mul_op(mat_val, x_val));
                    }
                }
            }
        }
    }

    _y->clear();
    for(auto it: map_output)
    {
        _y->push_back(it.first, it.second);
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmspv critical map BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
