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

    //cout << "C == " << C << endl;

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        ENT loc_cnt = 0;
        Y res_reg[VECTOR_LENGTH];
        #pragma _NEC vreg(res_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            res_reg[i] = 0;
        }

        #pragma _NEC novector
        #pragma omp for schedule(static, 8)
        for(VNT chunk=0; chunk < _matrix->nchunks; ++chunk)
        {
            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(VNT rowInChunk=0; rowInChunk < C; ++rowInChunk)
            {
                res_reg[rowInChunk] = op.identity();
            }

            loc_cnt += _matrix->chunkLen[chunk] * C;

            ENT idx = _matrix->chunkPtr[chunk];
            #pragma _NEC novector
            for(VNT j=0; j<_matrix->chunkLen[chunk]; j=j+P)
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #pragma _NEC gather_reorder
                for(VNT rowInChunk=0; rowInChunk<C; ++rowInChunk)
                {
                    if((chunk*C+rowInChunk) < _matrix->size)
                    {
                        A mat_val = _matrix->valSellC[idx+rowInChunk];
                        VNT col_id = _matrix->colSellC[idx+rowInChunk];
                        res_reg[rowInChunk] = add_op(res_reg[rowInChunk], mul_op(mat_val, x_vals[col_id])) ;
                    }
                }
                idx += C;
            }

            #pragma _NEC cncall
            #pragma _NEC ivdep
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(VNT rowInChunk=0; rowInChunk < C; ++rowInChunk)
            {
                buffer[chunk*C+rowInChunk] = res_reg[rowInChunk];
            }
        }

        /*#pragma omp critical
        {
            cout << " loc cnt: " << (100.0*loc_cnt) / _matrix->nnzSellC << endl;
        };*/
    }
    double t2 = omp_get_wtime();
    cout << "inner (cell c) time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "inner (cell c) BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    if(_matrix->sigma > 1)
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp parallel for
        for(VNT row = 0; row < _matrix->size; row++)
            y_vals[row] = _accum(y_vals[row], buffer[_matrix->sigmaInvPerm[row]]);
    }
    else
    {
        #pragma parallel omp for
        for(VNT row = 0; row < _matrix->size; row++)
        {
            y_vals[row] = _accum(y_vals[row], buffer[row]);
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////