#pragma once

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

template <typename T>
void SpMV(Matrix<T> &_matrix,
          Vector<T> &_x,
          Vector<T> &_y)
{
    SpMV(_matrix.csr_matrix, _x.dense, _y.dense);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////