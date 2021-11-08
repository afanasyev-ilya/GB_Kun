#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixCSR<T> &_A, DenseVector<T> &_x, DenseVector<T> &_y)
{
    VNT size = _A.get_size();
    VNT *row_ptr = _A.get_row_ptr();
    T *vals = _A.get_vals();
    ENT *col_ids = _A.get_col_ids();

    T *x_vals = _x.get_vals();
    T *y_vals = _y.get_vals();

    #pragma omp parallel for schedule(static)
    for(VNT i = 0; i < size; i++)
    {
        for(ENT j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            y_vals[i] += vals[j] * x_vals[col_ids[j]];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
