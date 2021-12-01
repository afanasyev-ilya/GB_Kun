#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
void SpMV(const MatrixSellC<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    cout << "in sellC spmv" << endl;

    /*#pragma omp parallel for schedule(static)
    for(VNT row = 0; row< _matrix->size; ++row)
    {
        T tmp = 0;
        for(VNT idx = _matrix->row_ptr[row]; idx < _matrix->row_ptr[row+1]; ++idx)
        {
            tmp += _matrix->vals[idx]*x_vals[_matrix->col_ids[idx]];
        }
        y_vals[row] = tmp;
    }*/

    const int C = _matrix->C;
    const int  P = _matrix->P;

    //#pragma omp parallel for schedule(static)
    for(int chunk=0; chunk<_matrix->nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            y_vals[chunk*C+rowInChunk] = 0;
        }
        for(int j=0; j<_matrix->chunkLen[chunk]; j=j+P)
        {
            int idx = _matrix->chunkPtr[chunk]+j*C;
            for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
            {
                y_vals[chunk*C+rowInChunk] += _matrix->valSellC[idx+rowInChunk]*x_vals[_matrix->colSellC[idx+rowInChunk]];
            }
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////