#pragma once
#ifdef __USE_KUNPENG__
#include <arm_neon.h>
#define INT_NEON_STRIDE 4
#define LONG_NEON_STRIDE 2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template<typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_all_active_diff_vectors_neon(const MatrixCSR <A> *_matrix,
                                  const DenseVector <X> *_x,
                                  DenseVector <Y> *_y,
                                  BinaryOpTAccum _accum,
                                  SemiringT op,
                                  Descriptor *_desc,
                                  Workspace *_workspace) {
    LOG_TRACE("Running SpMV_all_active_diff_vectors_NEON for CSR")
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    auto offsets = _matrix->get_load_balancing_offsets();

    #ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
    #endif
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    VNT first_row = offsets[tid].first;
    VNT last_row = offsets[tid].second;

    for(VNT row = first_row; row < last_row; row++)
    {
        /* Set all SIMD lanes to identity value */
        int64x2_t vec_resval = vmovq_n_s64(identity_val);

        Y res = identity_val;
        for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j+=LONG_NEON_STRIDE)
        {
            /* Perform scalar gather primitive, which is unavailable in NEON */
            VNT values[LONG_NEON_STRIDE] = {x_vals[_matrix->col_ids[j]], j + 1 ==  _matrix->row_ptr[row + 1] ? 0 : x_vals[_matrix->col_ids[j+1]]};

            /* Store vector and matrix values on vector registers */
            int64x2_t vec_cols = vld1q_s64((long int*)&values);

            int64x2_t vec_vals = vld1q_s64((long int*)&_matrix->vals[j]);

            int32x2_t cols_narrowed = vmovn_s64(vec_cols);
            int32x2_t vals_narrowed = vmovn_s64(vec_vals);


            /* Perform vector multiplication */

            /* 64-bit MUL for later work int64x2_t vec_mul =  arm_vmulq_s64(vec_vals, vec_cols); */
            int32x2_t vec_mul = vmul_s32(cols_narrowed, vals_narrowed);

            int64x2_t mul_widened = vmovl_s32(vec_mul);

            /* Sum of multplied pairs */
            vec_resval = vaddq_s64(mul_widened, vec_resval);
        }

        VNT temp_values[LONG_NEON_STRIDE] = {0, 0};
        vst1q_s64((long int*)&temp_values, vec_resval);

        y_vals[row] = _accum(y_vals[row], temp_values[0] + temp_values[1]);
    }
}
    #ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
    cout << "spmv slices (diff vector), unmasked time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

template<typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV_all_active_diff_vectors_neon_short(const MatrixCSR <A> *_matrix,
                                       const DenseVector <X> *_x,
                                       DenseVector <Y> *_y,
                                       BinaryOpTAccum _accum,
                                       SemiringT op,
                                       Descriptor *_desc,
                                       Workspace *_workspace) {
    LOG_TRACE("Running SpMV_all_active_diff_vectors_NEON_short for CSR")
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    auto offsets = _matrix->get_load_balancing_offsets();

#ifdef __DEBUG_BANDWIDTHS__
    double t1 = omp_get_wtime();
#endif
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VNT first_row = offsets[tid].first;
        VNT last_row = offsets[tid].second;

        for(VNT row = first_row; row < last_row; row++)
        {
            /* Set all SIMD lanes to identity value */
            int32x4_t vec_resval = vmovq_n_s32(identity_val);

            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j+=INT_NEON_STRIDE)
            {
                /* Perform scalar gather primitive, which is unavailable in NEON */
                int values[INT_NEON_STRIDE] = {x_vals[_matrix->col_ids[j]], j + 1 >=  _matrix->row_ptr[row + 1] ? 0 : x_vals[_matrix->col_ids[j+1]],
                                 j + 2 >=  _matrix->row_ptr[row + 1] ? 0 : x_vals[_matrix->col_ids[j+2]],
                                 j + 3 >=  _matrix->row_ptr[row + 1] ? 0 : x_vals[_matrix->col_ids[j+3]]};

                /* Store vector and matrix values on vector registers */
                int32x4_t vec_cols = vld1q_s32((int*)&values);
                int32x4_t vec_vals = vld1q_s32((int*)&_matrix->vals[j]);

                /* Perform vector multiplication */

                /* 64-bit MUL for later work int64x2_t vec_mul =  arm_vmulq_s64(vec_vals, vec_cols); */
                int32x4_t vec_mul = vmulq_s32(vec_cols, vec_vals);

                /* Sum of multplied pairs */
                vec_resval = vaddq_s32(vec_mul, vec_resval);
            }

            int temp_values[INT_NEON_STRIDE];
            vst1q_s32((int*)&temp_values, vec_resval);

            y_vals[row] = _accum(y_vals[row], temp_values[0] + temp_values[1] + temp_values[2] + temp_values[3]);
        }
    }
#ifdef __DEBUG_BANDWIDTHS__
    double t2 = omp_get_wtime();
cout << "spmv slices (diff vector), unmasked time: " << (t2 - t1)*1000 << " ms" << endl;
cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;
#endif
}
}
}
#endif