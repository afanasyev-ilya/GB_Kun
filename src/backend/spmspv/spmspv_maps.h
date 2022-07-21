#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_or_map(const MatrixCSR<A> *_matrix,
                            const SparseVector <X> *_x,
                            SparseVector <Y> *_y,
                            BinaryOpTAccum _accum,
                            SemiringT _op,
                            Descriptor *_desc,
                            Workspace *_workspace)
{
    LOG_TRACE("Running spmspv_unmasked_or_map")
    _matrix->apply_modifications();
    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();

    #ifdef __USE_TBB__
    tbb::concurrent_unordered_map<VNT, Y> map_output;
    #else
    std::unordered_map<VNT, Y> map_output;
    #endif

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
            ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
            ENT row_end   = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
                A mat_val = _matrix->vals[j];

                #ifdef __USE_TBB__
                map_output[dest_ind] = 1;
                #else
                #pragma omp critical
                {
                    map_output[dest_ind] = 1;
                }
                #endif
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
    cout << "spmspv or map BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_critical_map(const MatrixCSR<A> *_matrix,
                                  const SparseVector <X> *_x,
                                  SparseVector <Y> *_y,
                                  BinaryOpTAccum _accum,
                                  SemiringT _op,
                                  Descriptor *_desc,
                                  Workspace *_workspace)
{
    _matrix->apply_modifications();
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
            ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
            ENT row_end   = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
                A mat_val = _matrix->vals[j];

                #pragma omp critical
                {
                    /*if(map_output.find(dest_ind) == map_output.end())
                    {
                        map_output[dest_ind] = add_op(identity_val, mul_op(mat_val, x_val));
                    }
                    else
                    {
                        map_output[dest_ind] = add_op(map_output[dest_ind], mul_op(mat_val, x_val));
                    }*/
                    Y & search_res = map_output[dest_ind];
                    if(search_res == 0)
                        search_res = add_op(identity_val, mul_op(mat_val, x_val));
                    else
                        search_res = add_op(search_res, mul_op(mat_val, x_val));
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
