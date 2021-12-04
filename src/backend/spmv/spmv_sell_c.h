#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

#ifdef __ARM_FEATURE_SVE
template <typename T, typename SemiringT>
void SpMV(const MatrixSellC<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y, SemiringT op)
{
    /*const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    const uint64_t C_value = 32;
    if(_matrix->C != C_value)
    {
        printf("Wrong kernel this is SELL-%d\n", (int)C_value);
    }
    if(8 != svcntd())
    {
        printf("Wrong kernel this is 512-bit SIMD wide kernel\n");
    }

    #pragma omp parallel for schedule(static)
    for(int chunk=0; chunk<_matrix->nchunks; ++chunk)
    {
        uint64_t idx = 0;
        uint64_t base = _matrix->chunkPtr[chunk];
        uint64_t n = _matrix->chunkLen[chunk]*C_value;

        svfloat64_t tmp0; tmp0 = svadd_z(svpfalse(),tmp0, tmp0);
        svfloat64_t tmp1; tmp1 = svadd_z(svpfalse(),tmp1, tmp1);
        svfloat64_t tmp2; tmp2 = svadd_z(svpfalse(),tmp2, tmp2);
        svfloat64_t tmp3; tmp3 = svadd_z(svpfalse(),tmp3, tmp3);
        svbool_t pg = svwhilelt_b64(idx, n);
        double *base_val0 =  &(_matrix->valSellC[base+0*svcntd()]);
        double *base_val1 =  &(_matrix->valSellC[base+1*svcntd()]);
        double *base_val2 =  &(_matrix->valSellC[base+2*svcntd()]);
        double *base_val3 =  &(_matrix->valSellC[base+3*svcntd()]);
        int *base_col0 =  &(_matrix->colSellC[base+0*svcntd()]);
        int *base_col1 =  &(_matrix->colSellC[base+1*svcntd()]);
        int *base_col2 =  &(_matrix->colSellC[base+2*svcntd()]);
        int *base_col3 =  &(_matrix->colSellC[base+3*svcntd()]);

        do
        {
            svfloat64_t mat_val0 = svld1(pg, base_val0+idx);
            svfloat64_t mat_val1 = svld1(pg, base_val1+idx);
            svfloat64_t mat_val2 = svld1(pg, base_val2+idx);
            svfloat64_t mat_val3 = svld1(pg, base_val3+idx);
            svuint64_t mat_col0 = svld1sw_u64(pg, base_col0+idx);
            svuint64_t mat_col1 = svld1sw_u64(pg, base_col1+idx);
            svuint64_t mat_col2 = svld1sw_u64(pg, base_col2+idx);
            svuint64_t mat_col3 = svld1sw_u64(pg, base_col3+idx);
            svfloat64_t x_val0 = svld1_gather_index(pg, x_vals, mat_col0);
            svfloat64_t x_val1 = svld1_gather_index(pg, x_vals, mat_col1);
            svfloat64_t x_val2 = svld1_gather_index(pg, x_vals, mat_col2);
            svfloat64_t x_val3 = svld1_gather_index(pg, x_vals, mat_col3);
            tmp0 = svmla_m(pg, tmp0, mat_val0, x_val0);
            tmp1 = svmla_m(pg, tmp1, mat_val1, x_val1);
            tmp2 = svmla_m(pg, tmp2, mat_val2, x_val2);
            tmp3 = svmla_m(pg, tmp3, mat_val3, x_val3);
            idx += 4*svcntd();
            pg = svwhilelt_b64(idx, n);
        } while(svptest_any(svptrue_b64(), pg));

        svst1(svptrue_b64(), &(y_vals[C_value*chunk+0*svcntd()]), tmp0);
        svst1(svptrue_b64(), &(y_vals[C_value*chunk+1*svcntd()]), tmp1);
        svst1(svptrue_b64(), &(y_vals[C_value*chunk+2*svcntd()]), tmp2);
        svst1(svptrue_b64(), &(y_vals[C_value*chunk+3*svcntd()]), tmp3);
    }*/
}
#else
template <typename T, typename SemiringT>
void SpMV(const MatrixSellC<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y, SemiringT op)
{
    /*const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    const VNT C = _matrix->C;
    const VNT P = _matrix->P;

    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel for schedule(static)
    for(VNT chunk=0; chunk<_matrix->nchunks; ++chunk)
    {
        for(VNT rowInChunk=0; rowInChunk < C; ++rowInChunk)
        {
            if((chunk*C+rowInChunk) < _matrix->size)
                y_vals[chunk*C+rowInChunk] = op.identity();
        }

        for(VNT j=0; j<_matrix->chunkLen[chunk]; j=j+P)
        {
            ENT idx = _matrix->chunkPtr[chunk]+j*C;
            for(VNT rowInChunk=0; rowInChunk<C; ++rowInChunk)
            {
                if((chunk*C+rowInChunk) < _matrix->size)
                {
                    T mat_val = _matrix->valSellC[idx+rowInChunk];
                    VNT col_id = _matrix->colSellC[idx+rowInChunk];
                    y_vals[chunk*C+rowInChunk] = add_op(y_vals[chunk*C+rowInChunk], mul_op(mat_val, x_vals[col_id])) ;
                }
            }
        }
    }

    //cout << "cnt: " << cnt << " vs " << _matrix->nz << " " << (double) cnt / _matrix->nz << endl;*/
}
#endif

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////