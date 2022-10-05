/// @file spmspm_masked_clean.h
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Masked Clean SpMSpM algorithm
/// @details Implements Masked Clean IKJ SpMSpM algorithm
/// @date October 5, 2022

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @namespace Lablas
namespace lablas {

/// @namespace Backend
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Masked Clean SpMSpM algorithm.
///
/// @param[in] _result_mask Mask matrix
/// @param[in] _matrix1 Pointer to the first input matrix
/// @param[in] _matrix2 Pointer to the second input matrix
/// @param[out] _matrix_result Pointer to the (empty) matrix object that will contain the result matrix.
/// @param[in] _op Semiring operation
template <typename T, typename mask_type, typename SemiringT>
void SpMSpM_masked_clean(const Matrix<mask_type> *_result_mask,
                       const Matrix<T> *_matrix1,
                       const Matrix<T> *_matrix2,
                       Matrix<T> *_matrix_result,
                       SemiringT _op)
{
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    const auto n = _matrix1->get_csr()->get_num_rows();


    double t4 = omp_get_wtime();
    //SpMSpM_alloc(_matrix_result);
    //_matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);

    double t5 = omp_get_wtime();

    // Printing algorithm times:
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_masked_mxm_total_time", (t5 - t1) * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

