#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T, typename SemiringT>
void SpMV(const MatrixCOO<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    #pragma omp parallel for schedule(static)
    for(ENT i = 0; i < _matrix->nz; i++)
    {
        VNT row = _matrix->row_ids[i];
        VNT col = _matrix->col_ids[i];
        T val = _matrix->vals[i];
        #pragma omp atomic
        y_vals[row] += val * x_vals[col];
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
