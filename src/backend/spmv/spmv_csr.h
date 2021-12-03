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

template <typename T>
void SpMV(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    #pragma omp parallel
    {
        #pragma omp for schedule(guided, 1024)
        for(VNT i = 0; i < _matrix->size; i++)
        {
            for(ENT j = _matrix->row_ptr[i]; j < _matrix->row_ptr[i + 1]; j++)
            {
                y_vals[i] += _matrix->vals[j] * x_vals[_matrix->col_ids[j]];
            }
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////