/// @file spmspm_masked_hash_based_gustavson.h
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Masked hash-based IKJ SpMSpM algorithm
/// @details Implements Masked hash-based IKJ SpMSpM algorithm
/// @date June 8, 2022

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @namespace Lablas
namespace lablas {

/// @namespace Backend
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Masked Hash-based SpMSpM algorithm.
///
/// This algorithm is using IKJ-structured main loop that uses stl::unordered_map Hash Table to accumulate the
/// result matrix that is then exported to the CSR format. Algorithm assumes that result matrix will have the
/// same nnz structure as a mask matrix.
/// @param[in] _result_mask Mask matrix
/// @param[in] _matrix1 Pointer to the first input matrix
/// @param[in] _matrix2 Pointer to the second input matrix
/// @param[out] _matrix_result Pointer to the (empty) matrix object that will contain the result matrix.
/// @param[in] _op Semiring operation
template <typename T, typename mask_type, typename SemiringT>
void SpMSpM_masked_ikj(const Matrix<mask_type> *_result_mask,
                       const Matrix<T> *_matrix1,
                       const Matrix<T> *_matrix2,
                       Matrix<T> *_matrix_result,
                       SemiringT _op)
{
    LOG_TRACE("Running SpMSpM_masked_ikj")
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    auto mask_row_ptr = _result_mask->get_csr()->get_row_ptr();
    auto mask_col_ids_ptr = _result_mask->get_csr()->get_col_ids();

    // by assuming that result matrix will have same nnz structure as a mask, we could immediately
    // initialize all CSR columns for result matrix except for the vals column
    auto n = _result_mask->get_nrows();
    auto nnz = _result_mask->get_nnz();
    auto row_ptr = new ENT[n + 1];
    #pragma omp parallel for
    for (VNT i = 0; i <= n; ++i) {
        row_ptr[i] = mask_row_ptr[i];
    }
    auto col_ids = new VNT[nnz];
    #pragma omp parallel for
    for (VNT i = 0; i < nnz; ++i) {
        col_ids[i] = mask_col_ids_ptr[i];
    }
    auto vals = new T[nnz];
    #pragma omp parallel for
    for (VNT i = 0; i < nnz; ++i) {
        vals[i] = 0;
    }

    auto offsets = _result_mask->get_csr()->get_load_balancing_offsets();

    MatrixCSR<T> A_csr_first_socket;
    MatrixCSR<T> A_csr_second_socket;
    MatrixCSR<T> B_csr_first_socket;
    MatrixCSR<T> B_csr_second_socket;
    if (num_sockets_used() == 2) {
        LOG_TRACE("Using NUMA optimization")
        A_csr_first_socket.deep_copy(_matrix1->get_csr(), 0);
        A_csr_second_socket.deep_copy(_matrix1->get_csr(), 1);
        B_csr_first_socket.deep_copy(_matrix2->get_csr(), 0);
        B_csr_second_socket.deep_copy(_matrix2->get_csr(), 1);
    }

    #ifdef __USE_KUNPENG__
        const int max_threads_per_socket = sysconf(_SC_NPROCESSORS_ONLN)/2;
    #else
        const int max_threads_per_socket = omp_get_max_threads();
    #endif

    double t2 = omp_get_wtime();

    // main parallel IKJ loop
    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();

        // explicit optimization for saving pointer's values on stack variables
        const ENT * matrix1_row_ptr;
        const VNT * matrix1_col_ids_ptr;
        const T * matrix1_vals_ptr;
        const ENT * matrix2_row_ptr;
        const VNT * matrix2_col_ids_ptr;
        const T * matrix2_vals_ptr;

        if (num_sockets_used() == 2) {
            // In this case using NUMA optimization and copying input matrices on both sockets
            #ifdef __USE_KUNPENG__
                const int cpu_id = sched_getcpu();
            #else
                const int cpu_id = thread_id;
            #endif
            const int socket = cpu_id / (max_threads_per_socket);

            if (socket == 0) {
                matrix1_row_ptr = A_csr_first_socket.get_row_ptr();
                matrix1_col_ids_ptr = A_csr_first_socket.get_col_ids();
                matrix1_vals_ptr = A_csr_first_socket.get_vals();
                matrix2_row_ptr = B_csr_first_socket.get_row_ptr();
                matrix2_col_ids_ptr = B_csr_first_socket.get_col_ids();
                matrix2_vals_ptr = B_csr_first_socket.get_vals();
            } else {
                matrix1_row_ptr = A_csr_second_socket.get_row_ptr();
                matrix1_col_ids_ptr = A_csr_second_socket.get_col_ids();
                matrix1_vals_ptr = A_csr_second_socket.get_vals();
                matrix2_row_ptr = B_csr_second_socket.get_row_ptr();
                matrix2_col_ids_ptr = B_csr_second_socket.get_col_ids();
                matrix2_vals_ptr = B_csr_second_socket.get_vals();
            }
        } else {
            matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
            matrix1_col_ids_ptr = _matrix1->get_csr()->get_col_ids();
            matrix1_vals_ptr = _matrix1->get_csr()->get_vals();
            matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
            matrix2_col_ids_ptr = _matrix2->get_csr()->get_col_ids();
            matrix2_vals_ptr = _matrix2->get_csr()->get_vals();
        }

        // i-th loop
        for (VNT matrix1_row_id = offsets[thread_id].first; matrix1_row_id < offsets[thread_id].second;
             ++matrix1_row_id) {
            // for each row we will contain its own set of used mask column ids
            // and will accumulate result values in its own hash map
            unordered_set<ENT> mask_col_ids_set;
            unordered_map<VNT, T> matrix_result;
            for (VNT mask_row_ptr_id = mask_row_ptr[matrix1_row_id];
                 mask_row_ptr_id < mask_row_ptr[matrix1_row_id + 1]; ++mask_row_ptr_id) {
                mask_col_ids_set.insert(mask_col_ids_ptr[mask_row_ptr_id]);
            }

            // k-th loop
            for (VNT matrix1_row_ptr_id = matrix1_row_ptr[matrix1_row_id];
                 matrix1_row_ptr_id < matrix1_row_ptr[matrix1_row_id + 1]; ++matrix1_row_ptr_id) {
                VNT k = matrix1_col_ids_ptr[matrix1_row_ptr_id];
                // j-th loop
                for (VNT matrix2_col_id = matrix2_row_ptr[k];
                     matrix2_col_id < matrix2_row_ptr[k + 1]; ++matrix2_col_id) {
                    VNT j = matrix2_col_ids_ptr[matrix2_col_id];
                    if (mask_col_ids_set.find(j) == mask_col_ids_set.end()) {
                        continue;
                    }
                    if (matrix_result.find(j) == matrix_result.end()) {
                        matrix_result[j] = identity_val;
                    }
                    matrix_result[j] = add_op(matrix_result[j],
                            mul_op(matrix1_vals_ptr[matrix1_row_ptr_id], matrix2_vals_ptr[matrix2_col_id]));
                }
            }
            auto vals_id = row_ptr[matrix1_row_id];
            for (const auto & [key, value] : matrix_result) {
                vals[vals_id] = value;
                ++vals_id;
            }
        }
    }

    double t3 = omp_get_wtime();

    // Building result matrix from CSR arrays
    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);
    double t4 = omp_get_wtime();

    double overall_time = t4 - t1;

    // saving performance results into the output file
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ikj_masked_preparation", (t2-t1) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ikj_mxm_inner_loop", (t3-t2) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ikj_mxm_csr_export", (t4-t3) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ikj_masked_mxm", overall_time * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

    #ifdef __DEBUG_INFO__
        printf("Masked IKJ SpMSpM time: %lf seconds.\n", overall_time);
        printf("\t- Preparing data before evaluations: %.1lf %%\n", (t2 - t1) / overall_time * 100.0);
        printf("\t- Main IKJ loop: %.1lf %%\n", (t3 - t2) / overall_time * 100.0);
        printf("\t- Converting CSR result to Matrix object: %.1lf %%\n", (t4 - t3) / overall_time * 100.0);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
