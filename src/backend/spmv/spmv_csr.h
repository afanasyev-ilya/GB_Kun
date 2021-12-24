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

template <typename T, typename SemiringT, typename BinaryOpTAccum>
void SpMV_numa_aware(MatrixCSR<T> *_matrix,
                     MatrixCSR<T> *_matrix_socket_dub,
                     const DenseVector<T> *_x,
                     DenseVector<T> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *x_vals_first_socket = (T*)_matrix->tmp_buffer;
    T *x_vals_second_socket = (T*)_matrix_socket_dub->tmp_buffer;

    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #pragma omp parallel
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
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT, typename BinaryOpTAccum>
void SpMV_non_optimized(MatrixCSR<T> *_matrix,
                        const DenseVector<T> *_x,
                        DenseVector<T> *_y,
                        BinaryOpTAccum _accum,
                        SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT row = 0; row < _matrix->size; row++)
        {
            T res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                T val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CMP, typename T, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_dense(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y,
          BinaryOpTAccum _accum,
          SemiringT op,
          const Vector<Y> *_mask)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix->size; i++)
        {
            bool mask_val = (bool)_mask->getDense()->get_vals()[i];
            if (mask_val && !CMP) {
                T res = identity_val;
                for(ENT j = _matrix->row_ptr[i]; j < _matrix->row_ptr[i + 1]; j++)
                {
                    res = add_op(res, mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                }
                y_vals[i] = _accum(y_vals, res);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT, typename BinaryOpTAccum>
void SpMV(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y,
          BinaryOpTAccum _accum,
          SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix->vg_num; vg++)
        {
            const VNT *vertices = _matrix->vertex_groups[vg].get_data();
            VNT vertex_group_size = _matrix->vertex_groups[vg].get_size();

            #pragma omp for nowait schedule(guided, 1)
            for(VNT idx = 0; idx < vertex_group_size; idx++)
            {
                VNT row = vertices[idx];
                T res = identity_val;
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    VNT col = _matrix->col_ids[j];
                    T val = _matrix->vals[j];
                    res = add_op(res, mul_op(val, x_vals[col]));
                }
                y_vals[row] = _accum(res, y_vals[row]);
            }
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////