#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
void SpMV(const MatrixSellC<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          BinaryOpTAccum _accum,
          SemiringT op,
          Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();

    const VNT C = _matrix->C;
    const VNT P = _matrix->P;

    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    Y *buffer = (Y*)_workspace->get_first_socket_vector();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT chunk=0; chunk < _matrix->nchunks; ++chunk)
        {
            for(VNT rowInChunk=0; rowInChunk < C; ++rowInChunk)
            {
                if((chunk*C+rowInChunk) < _matrix->size)
                    buffer[chunk*C+rowInChunk] = op.identity();
            }

            for(VNT j=0; j<_matrix->chunkLen[chunk]; j=j+P)
            {
                ENT idx = _matrix->chunkPtr[chunk]+j*C;
                for(VNT rowInChunk=0; rowInChunk<C; ++rowInChunk)
                {
                    if((chunk*C+rowInChunk) < _matrix->size)
                    {
                        A mat_val = _matrix->valSellC[idx+rowInChunk];
                        VNT col_id = _matrix->colSellC[idx+rowInChunk];
                        buffer[chunk*C+rowInChunk] = add_op(buffer[chunk*C+rowInChunk], mul_op(mat_val, x_vals[col_id])) ;
                    }
                }
            }
        }

        #pragma omp for
        for(VNT row = 0; row < _matrix->size; row++)
        {
            y_vals[row] = _accum(y_vals[row], buffer[row]);
        }
    }

}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////