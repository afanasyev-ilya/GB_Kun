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

    const int C = _matrix->C;
    const int  P = _matrix->P;

    #pragma omp parallel for schedule(static)
    for(int chunk=0; chunk<_matrix->nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk < C; ++rowInChunk)
        {
            if((chunk*C+rowInChunk) < _matrix->size)
                y_vals[chunk*C+rowInChunk] = 0;
        }

        for(int j=0; j<_matrix->chunkLen[chunk]; j=j+P)
        {
            int idx = _matrix->chunkPtr[chunk]+j*C;
            for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
            {
                if((chunk*C+rowInChunk) < _matrix->size)
                {
                    T mat_val = _matrix->valSellC[idx+rowInChunk];
                    VNT col_id = _matrix->colSellC[idx+rowInChunk];
                    y_vals[chunk*C+rowInChunk] += mat_val * x_vals[col_id];
                }
            }
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////