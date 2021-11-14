#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixCSR<T> &_matrix,
          DenseVector<T> &_x,
          DenseVector<T> &_y)
{

    #pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int socket = tid/(total_threads/2);
        MatrixCSR<T> &_local_matrix = _matrix;

        #pragma omp for schedule(static)
        for(VNT i = 0; i < _local_matrix.size; i++)
        {
            for(ENT j = _local_matrix.row_ptr[i]; j < _local_matrix.row_ptr[i + 1]; j++)
            {
                _y.vals[i] += _local_matrix.vals[j] * _x.vals[_local_matrix.col_ids[j]];
            }
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////