#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/*template <typename T>
void SpMV(MatrixCSR<T> &_matrix,
          MatrixCSR<T> &_matrix_socket_dub,
          DenseVector<T> &_x,
          DenseVector<T> &_y,
          Descriptor &_desc)
{
    #pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int socket = tid/(total_threads/2);
        int tid_s = omp_get_thread_num() % (total_threads/2);

        T* local_buffer = (T*)_desc->tmp_buffer;

        MatrixCSR<T> &local_matrix = _matrix;
        if(socket == 0)
        {
            local_matrix = _matrix;
            local_buffer = _x->vals;
        }
        else if(socket == 1)
        {
            local_matrix = _matrix_socket_dub;
        }

        for(VNT i = tid_s; i < _matrix->size; i += total_threads/2)
            local_buffer[i] = _x->vals[i];

        #pragma omp for schedule(static)
        for(VNT i = 0; i < local_matrix->size; i++)
        {
            for(ENT j = local_matrix->row_ptr[i]; j < local_matrix->row_ptr[i + 1]; j++)
            {
                _y->vals[i] += local_matrix->vals[j] * _x->vals[local_matrix->col_ids[j]];
            }
        }
    };
}*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CMP, typename T, typename Y, typename SemiringT>
void SpMV(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y, SemiringT op, const Vector<Y> *_mask)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(VNT i = 0; i < _matrix->size; i++)
        {
            bool mask_val = (bool)_mask->getDense()->get_vals()[i];
            if (mask_val && !CMP) {
                for(ENT j = _matrix->row_ptr[i]; j < _matrix->row_ptr[i + 1]; j++)
                {
                    y_vals[i] = add_op(y_vals[i], mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                }
            }
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////