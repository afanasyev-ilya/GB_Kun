#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void in_socket_copy(T* _local_data, const T *_shared_data, VNT _size, int _max_threads_per_socket)
{
    int tid = omp_get_thread_num() % _max_threads_per_socket;
    VNT work_per_thread = (_size - 1) / _max_threads_per_socket + 1;

    for(VNT i = min(_size, tid*work_per_thread); i < min(_size, (tid + 1)*work_per_thread); i++)
    {
        _local_data[i] = _shared_data[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_numa_aware(const MatrixCSR<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    X *x_vals_first_socket = (X*)_workspace->get_first_socket_vector();
    X *x_vals_second_socket = (X*)_workspace->get_second_socket_vector();

    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    auto offsets = _matrix->get_load_balancing_offsets();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif

    #ifdef __USE_KUNPENG__
    const int max_threads_per_socket = sysconf(_SC_NPROCESSORS_ONLN)/2;
    #else
    const int max_threads_per_socket = omp_get_max_threads();
    #endif

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #ifdef __USE_KUNPENG__
        const int cpu_id = sched_getcpu();
        #else
        const int cpu_id = tid;
        #endif

        const int socket = cpu_id / (max_threads_per_socket);

        X *local_x_vals = 0;
        const int total_threads = omp_get_num_threads();
        if(total_threads == max_threads_per_socket*2) // if 96 or 128 threads
        {
            if(socket == 0) // we use in socket copy, which works only on max_threads_per_socket threads
            {
                local_x_vals = x_vals_first_socket;
                in_socket_copy(local_x_vals, x_vals, _matrix->nrows, max_threads_per_socket);
            }
            else if(socket == 1)
            {
                local_x_vals = x_vals_second_socket;
                in_socket_copy(local_x_vals, x_vals, _matrix->nrows, max_threads_per_socket);
            }
        }
        else
        {
            #pragma omp for
            for(VNT i = 0; i < _matrix->nrows; i++)
            {
                x_vals_first_socket[i] = x_vals[i];
                x_vals_second_socket[i] = x_vals[i];
            }

            if(socket == 0)
            {
                local_x_vals = x_vals_first_socket;
            }
            else if(socket == 1)
            {
                local_x_vals = x_vals_second_socket;
            }
        }

        VNT first_row = offsets[tid].first;
        VNT last_row = offsets[tid].second;

        ENT *shifts = _matrix->row_ptr;
        VNT *connections_count = _matrix->row_degrees;
        VNT *col_ids = _matrix->col_ids;
        A *vals = _matrix->vals;

        for(VNT row = first_row; row < last_row; row++)
        {
            Y res = identity_val;
            ENT shift = shifts[row];
            VNT connections = connections_count[row];
            for(ENT j = shift; j < shift + connections; j++)
            {
                VNT col = col_ids[j];
                A val = vals[j];
                res = add_op(res, mul_op(val, local_x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv slices (numa-aware), unmasked time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_all_active_static(const MatrixCSR<A> *_matrix,
                            const DenseVector<X> *_x,
                            DenseVector<Y> *_y,
                            BinaryOpTAccum _accum,
                            SemiringT op,
                            Descriptor *_desc,
                            Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT row = 0; row < _matrix->nrows; row++)
        {
            A res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMV_sparse(const MatrixCSR<A> *_matrix,
                 const DenseVector<X> *_x,
                 DenseVector<Y> *_y,
                 BinaryOpTAccum _accum,
                 SemiringT op,
                 const SparseVector<M> *_mask,
                 Descriptor *_desc,
                 Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif

    const VNT mask_nvals = _mask->get_nvals();
    const VNT *mask_ids = _mask->get_ids();
    const M* mask_vals = _mask->get_vals();

    Desc_value mask_field;
    _desc->get(GrB_MASK, &mask_field);
    if (mask_field != GrB_STR_COMP)
    {
        #pragma omp parallel
        {
            #pragma omp for schedule(guided, 1)
            for(VNT i = 0; i < mask_nvals; i++)
            {
                VNT row = mask_ids[i];

                Y res = identity_val;
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    res = add_op(res, mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                }
                y_vals[row] = _accum(y_vals[row], res);
            }
        }
    }
    else
    {
        Index *dense_mask = _workspace->get_mask_conversion();

        #pragma omp parallel
        {
            #pragma omp for
            for(VNT i = 0; i < _matrix->nrows; i++)
                dense_mask[i] = 0;

            #pragma omp for
            for(VNT i = 0; i < mask_nvals; i++)
            {
                VNT row = mask_ids[i];
                dense_mask[row] = 1;
            }

            #pragma omp for schedule(guided, 1)
            for(VNT row = 0; row < _matrix->nrows; row++)
            {
                bool mask_val = dense_mask[row];
                if(mask_val == 0)
                {
                    Y res = identity_val;
                    for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                    {
                        res = add_op(res, mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                    }
                    y_vals[row] = _accum(y_vals[row], res);
                }
                else
                {
                    y_vals[row] = 0; // if assign
                }
            }
        }
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv sparse: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMV_dense(const MatrixCSR<A> *_matrix,
                const DenseVector<X> *_x,
                DenseVector<Y> *_y,
                BinaryOpTAccum _accum,
                SemiringT op,
                const DenseVector<M> *_mask,
                Descriptor *_desc,
                Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif

    /*if(x_vals == y_vals)
    {
        X *tmp_vals = (X*)_workspace->get_first_socket_vector();
        #pragma omp parallel for
        for(VNT i = 0; i < _matrix->size; i++)
            tmp_vals[i] = x_vals[i];
        x_vals = tmp_vals;
    }*/

    const M *mask_vals = _mask->get_vals();
    bool use_comp_mask;

    Desc_value mask_field;
    _desc->get(GrB_MASK, &mask_field);
    if (mask_field == GrB_STR_COMP)
    {
        use_comp_mask = true;
    }
    else
    {
        use_comp_mask = false;
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(guided, 1)
        for(VNT row = 0; row < _matrix->nrows; row++)
        {
            bool mask_val = (bool)mask_vals[row];
            if (!use_comp_mask && mask_val || use_comp_mask && !mask_val)
            {
                Y res = identity_val;
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    res = add_op(res, mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                }
                y_vals[row] = _accum(y_vals[row], res);
            }
            else
            {
                y_vals[row] = 0; // if assign
            }
        }
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv dense: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_all_active_diff_vectors(const MatrixCSR<A> *_matrix,
                                  const DenseVector<X> *_x,
                                  DenseVector<Y> *_y,
                                  BinaryOpTAccum _accum,
                                  SemiringT op,
                                  Descriptor *_desc,
                                  Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    auto offsets = _matrix->get_load_balancing_offsets();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VNT first_row = offsets[tid].first;
        VNT last_row = offsets[tid].second;

        for(VNT row = first_row; row < last_row; row++)
        {
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    }
    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv slices (diff vector), unmasked time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif

    /*t1 = omp_get_wtime();
    tbb::parallel_for( tbb::blocked_range<int>(0, _matrix->nrows),
                       [&](tbb::blocked_range<int> r)
    {
       for (int row=r.begin(); row<r.end(); ++row)
       {
           Y res = identity_val;
           for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
           {
               VNT col = _matrix->col_ids[j];
               A val = _matrix->vals[j];
               res = add_op(res, mul_op(val, x_vals[col]));
           }
           y_vals[row] = _accum(y_vals[row], res);
       }
    }, tbb::static_partitioner());
    t2 = omp_get_wtime();
    cout << "tbb spmv bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_all_active_same_vectors(const MatrixCSR<A> *_matrix,
                                  const DenseVector<X> *_x,
                                  DenseVector<Y> *_y,
                                  BinaryOpTAccum _accum,
                                  SemiringT op,
                                  Descriptor *_desc,
                                  Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    auto offsets = _matrix->get_load_balancing_offsets();
    Y *buffer = (Y*)_workspace->get_first_socket_vector();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VNT first_row = offsets[tid].first;
        VNT last_row = offsets[tid].second;

        for(VNT row = first_row; row < last_row; row++)
        {
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            buffer[row] = res;
        }

        #pragma omp barrier

        #pragma omp for
        for(VNT row = 0; row < _matrix->nrows; row++)
        {
            y_vals[row] = _accum(y_vals[row], buffer[row]);
        }
    }

    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv slices (same vector), unmasked time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
