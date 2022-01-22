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
            if(!_matrix->problematic_chunk[chunk]) // if usual Sell-C chunk, use Sell-C processing
            {
                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vector
                #pragma _NEC gather_reorder
                for(VNT row_in_chunk=0; row_in_chunk < C; ++row_in_chunk)
                {
                    res_reg[row_in_chunk] = op.identity();
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
                    for(VNT row_in_chunk=0; row_in_chunk<C; ++row_in_chunk)
                    {
                        if((chunk*C+row_in_chunk) < _matrix->size)
                        {
                            A mat_val = _matrix->valSellC[idx+row_in_chunk];
                            VNT col_id = _matrix->colSellC[idx+row_in_chunk];
                            res_reg[row_in_chunk] = add_op(res_reg[row_in_chunk], mul_op(mat_val, x_vals[col_id])) ;
                        }
                    }
                    idx += C;
                }

                #pragma _NEC cncall
                #pragma _NEC ivdep
                #pragma _NEC vector
                #pragma _NEC gather_reorder
                for(VNT row_in_chunk=0; row_in_chunk < C; ++row_in_chunk)
                {
                    buffer[chunk*C+row_in_chunk] = res_reg[row_in_chunk];
                }
            }
            else // if problematic Sell-C chunk, use CSR processing
            {
                #pragma _NEC novector
                for(VNT row_in_chunk = 0; row_in_chunk < C; ++row_in_chunk)
                {
                    Y res= identity_val;
                    VNT row = chunk*C+row_in_chunk;
                    #pragma _NEC cncall
                    #pragma _NEC ivdep
                    #pragma _NEC vovertake
                    #pragma _NEC novob
                    #pragma _NEC vector
                    #pragma _NEC gather_reorder
                    for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                    {
                        VNT col = _matrix->col_ids[j];
                        X val = _matrix->vals[j];
                        res = add_op(res, mul_op(val, x_vals[col]));
                    }
                    buffer[row] = res;
                }
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

    t1 = omp_get_wtime();
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
    t2 = omp_get_wtime();
    cout << "reorder (cell c) time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "reorder (cell c) BW: " << _matrix->size * (4.0*sizeof(X) + sizeof(VNT)) / ((t2 - t1)*1e9) << " GB/s" << endl;
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
