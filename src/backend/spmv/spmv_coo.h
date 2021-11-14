#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixCOO<T> &_matrix, DenseVector<T> &_x, DenseVector<T> &_y)
{
    #pragma omp parallel for schedule(static)
    for(ENT i = 0; i < _matrix.nz; i++)
    {
        VNT row = _matrix.row_ids[i];
        VNT col = _matrix.col_ids[i];
        T val = _matrix.vals[i];
        #pragma omp atomic
        _y.vals[row] += val * _x.vals[col];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
