#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
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

        T* local_buffer = (T*)_desc.tmp_buffer;
        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix.size; i++)
            local_buffer[i] = _x.vals[i];

        MatrixCSR<T> &local_matrix = _matrix;
        if(socket == 0)
        {
            local_matrix = _matrix;
            local_buffer = _x.vals;
        }
        else if(socket == 1)
        {
            local_matrix = _matrix_socket_dub;
        }

        #pragma omp for schedule(static)
        for(VNT i = 0; i < local_matrix.size; i++)
        {
            for(ENT j = local_matrix.row_ptr[i]; j < local_matrix.row_ptr[i + 1]; j++)
            {
                _y.vals[i] += local_matrix.vals[j] * local_buffer[local_matrix.col_ids[j]];
            }
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixCSR<T> &_matrix,
          DenseVector<T> &_x,
          DenseVector<T> &_y)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix.size; i++)
        {
            for(ENT j = _matrix.row_ptr[i]; j < _matrix.row_ptr[i + 1]; j++)
            {
                _y.vals[i] += _matrix.vals[j] * _x.vals[_matrix.col_ids[j]];
            }
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////