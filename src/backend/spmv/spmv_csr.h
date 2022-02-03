#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
void in_socket_copy(T* _local_data, const T *_shared_data, VNT _size)
{
    int tid = omp_get_thread_num() % THREADS_PER_SOCKET;
    VNT work_per_thread = (_size - 1) / THREADS_PER_SOCKET + 1;

    for(VNT i = tid*work_per_thread; i < min(_size, (tid + 1)*work_per_thread); i++)
    {
        _local_data[i] = _shared_data[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_numa_aware(MatrixCSR<A> *_matrix,
                     MatrixCSR<A> *_matrix_socket_dub,
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

    /*#pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int socket = tid / (THREADS_PER_SOCKET);

        T *local_x_vals, *local_y_vals;

        VNT vec_size = _matrix->size;
        MatrixCSR<T> *local_matrix;
        if(socket == 0)
        {
            local_x_vals = x_vals_first_socket;
            local_matrix = _matrix;
            in_socket_copy(local_x_vals, x_vals, vec_size);
        }
        else if(socket == 1)
        {
            local_x_vals = x_vals_second_socket;
            local_matrix = _matrix_socket_dub;
            in_socket_copy(local_x_vals, x_vals, vec_size);
        }

        #pragma omp barrier

        for(int vg = 0; vg < _matrix->vg_num; vg++)
        {
            const VNT *vertices = _matrix->vertex_groups[vg].get_data();
            VNT vertex_group_size = _matrix->vertex_groups[vg].get_size();

            #pragma omp for nowait schedule(guided, 1)
            for(VNT idx = 0; idx < vertex_group_size; idx++)
            {
                VNT row = vertices[idx];
                T res = identity_val;
                for(ENT j = local_matrix->row_ptr[row]; j < local_matrix->row_ptr[row + 1]; j++)
                {
                    VNT col = local_matrix->col_ids[j];
                    T val = local_matrix->vals[j];
                    res = add_op(res, mul_op(val, local_x_vals[col]));
                }
                y_vals[row] = _accum(y_vals[row], res);
            }
        }
    }*/

    #pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int socket = tid / (THREADS_PER_SOCKET);

        X *local_x_vals = 0;
        if(socket == 0)
        {
            local_x_vals = x_vals_first_socket;
            in_socket_copy(local_x_vals, x_vals, _matrix->size);
        }
        else if(socket == 1)
        {
            local_x_vals = x_vals_second_socket;
            in_socket_copy(local_x_vals, x_vals, _matrix->size);
        }

        #pragma omp for nowait schedule(static, 1)
        for(VNT i = 0; i < _matrix->large_degree_threshold; i++)
        {
            VNT row = _matrix->sorted_rows[i];
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, local_x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }

        #pragma omp for nowait schedule(static, CSR_SORTED_BALANCING)
        for(VNT i = _matrix->large_degree_threshold; i < _matrix->size; i++)
        {
            VNT row = _matrix->sorted_rows[i];
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, local_x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    }
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
        for(VNT row = 0; row < _matrix->size; row++)
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
            for(VNT i = 0; i < _matrix->size; i++)
                dense_mask[i] = 1;

            #pragma omp for
            for(VNT i = 0; i < mask_nvals; i++)
            {
                VNT row = mask_ids[i];
                dense_mask[row] = 0;
            }

            #pragma omp for schedule(guided, 1)
            for(VNT row = 0; row < _matrix->size; row++)
            {
                bool mask_val = dense_mask[row];
                if(mask_val)
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
        for(VNT row = 0; row < _matrix->size; row++)
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

    #pragma omp parallel
    {
        #pragma omp for nowait schedule(static, 1)
        for(VNT i = 0; i < _matrix->large_degree_threshold; i++)
        {
            VNT row = _matrix->sorted_rows[i];
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }

        #pragma omp for nowait schedule(static, CSR_SORTED_BALANCING)
        for(VNT i = _matrix->large_degree_threshold; i < _matrix->size; i++)
        {
            VNT row = _matrix->sorted_rows[i];
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

    Y *buffer = (Y*)_workspace->get_first_socket_vector();

    #pragma omp parallel
    {
        #pragma omp for nowait schedule(static, 1)
        for(VNT i = 0; i < _matrix->large_degree_threshold; i++)
        {
            VNT row = _matrix->sorted_rows[i];
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            buffer[row] = res;
        }

        #pragma omp for nowait schedule(static, CSR_SORTED_BALANCING)
        for(VNT i = _matrix->large_degree_threshold; i < _matrix->size; i++)
        {
            VNT row = _matrix->sorted_rows[i];
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
        for(VNT row = 0; row < _matrix->size; row++)
        {
            y_vals[row] = _accum(y_vals[row], buffer[row]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
