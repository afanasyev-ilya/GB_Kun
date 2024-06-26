#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VNT spgemm_lower_bound(const Index* data, VNT left, VNT right, ENT value)
{
    while (left < right) {
        VNT mid =  left + (right - left) / 2;
        if (value <= data[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void SpMSpM_ijk(const Matrix<T> *_matrix1,
                const Matrix<T> *_matrix2,
                Matrix<T> *_matrix_result,
                const Matrix<T> *_result_mask,
                SemiringT _op,
                bool a_is_sorted)
{
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    auto mask_row_ptr = _result_mask->get_csr()->get_row_ptr();
    auto mask_col_ids_ptr = _result_mask->get_csr()->get_col_ids();
    auto mask_vals_ptr = _result_mask->get_csr()->get_vals();

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

    auto offsets = _result_mask->get_csr()->get_load_balancing_offsets();

    MatrixCSR<T> A_csr_first_socket;
    MatrixCSR<T> A_csr_second_socket;
    MatrixCSR<T> B_csc_first_socket;
    MatrixCSR<T> B_csc_second_socket;
    if (num_sockets_used() == 2) {
        #ifdef __DEBUG_INFO__
            cout << "Using NUMA optimization" << endl;
        #endif
        A_csr_first_socket.deep_copy(_matrix1->get_csr(), 0);
        A_csr_second_socket.deep_copy(_matrix1->get_csr(), 1);
        B_csc_first_socket.deep_copy(_matrix2->get_csc(), 0);
        B_csc_second_socket.deep_copy(_matrix2->get_csc(), 1);
    }

    #ifdef __USE_KUNPENG__
        const int max_threads_per_socket = sysconf(_SC_NPROCESSORS_ONLN)/2;
    #else
        const int max_threads_per_socket = omp_get_max_threads();
    #endif

    double t2 = omp_get_wtime();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();

        const ENT * matrix1_row_ptr;
        const VNT * matrix1_col_ids_ptr;
        const T * matrix1_vals_ptr;
        const ENT * matrix2_row_ptr;
        const VNT * matrix2_col_ids_ptr;
        const T * matrix2_vals_ptr;

        if (num_sockets_used() == 2) {
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
                matrix2_row_ptr = B_csc_first_socket.get_row_ptr();
                matrix2_col_ids_ptr = B_csc_first_socket.get_col_ids();
                matrix2_vals_ptr = B_csc_first_socket.get_vals();
            } else {
                matrix1_row_ptr = A_csr_second_socket.get_row_ptr();
                matrix1_col_ids_ptr = A_csr_second_socket.get_col_ids();
                matrix1_vals_ptr = A_csr_second_socket.get_vals();
                matrix2_row_ptr = B_csc_second_socket.get_row_ptr();
                matrix2_col_ids_ptr = B_csc_second_socket.get_col_ids();
                matrix2_vals_ptr = B_csc_second_socket.get_vals();
            }
        } else {
            matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
            matrix1_col_ids_ptr = _matrix1->get_csr()->get_col_ids();
            matrix1_vals_ptr = _matrix1->get_csr()->get_vals();
            matrix2_row_ptr = _matrix2->get_csc()->get_row_ptr();
            matrix2_col_ids_ptr = _matrix2->get_csc()->get_col_ids();
            matrix2_vals_ptr = _matrix2->get_csc()->get_vals();
        }
        for (VNT matrix1_row_id = offsets[thread_id].first; matrix1_row_id < offsets[thread_id].second;
             ++matrix1_row_id) {
            ENT matrix1_col_start_id = matrix1_row_ptr[matrix1_row_id];
            ENT matrix1_col_end_id = matrix1_row_ptr[matrix1_row_id + 1];
            ENT mask_col_start_id = mask_row_ptr[matrix1_row_id];
            ENT mask_col_end_id = mask_row_ptr[matrix1_row_id + 1];
            for (ENT mask_col_id = mask_col_start_id; mask_col_id < mask_col_end_id; ++mask_col_id) {
                if (!mask_vals_ptr[mask_col_id]) {
                    continue;
                }
                VNT matrix2_col_id = mask_col_ids_ptr[mask_col_id];
                VNT matrix2_col_start_id = matrix2_row_ptr[matrix2_col_id];
                VNT matrix2_col_end_id = matrix2_row_ptr[matrix2_col_id + 1];

                T accumulator = identity_val;

                VNT matrix2_row_id = matrix2_col_start_id;

                for (ENT matrix1_col_id = matrix1_col_start_id;
                     matrix1_col_id < matrix1_col_end_id; ++matrix1_col_id) {
                    ENT matrix1_col_num = matrix1_col_ids_ptr[matrix1_col_id];
                    // i == matrix1_row_id, j == matrix2_col_id, k == matrix1_col_num
                    VNT found_matrix2_row_id = spgemm_lower_bound(matrix2_col_ids_ptr,
                                                                  matrix2_row_id,
                                                                  matrix2_col_end_id,
                                                                  matrix1_col_num);

                    if (found_matrix2_row_id == matrix2_col_end_id) {
                        break;
                    }
                    if (a_is_sorted) {
                        matrix2_row_id = found_matrix2_row_id;
                    }

                    if (matrix2_col_ids_ptr[found_matrix2_row_id] == matrix1_col_num) {
                        accumulator = add_op(accumulator,
                                             mul_op(matrix1_vals_ptr[matrix1_col_id],
                                                    matrix2_vals_ptr[found_matrix2_row_id]));
                    }
                }
                vals[mask_col_id] = accumulator;
            }
        }
    }

    double t3 = omp_get_wtime();

    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);
    double t4 = omp_get_wtime();

    double overall_time = t4 - t1;

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ijk_masked_mxm", overall_time * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

    #ifdef __DEBUG_INFO__
        printf("Unmasked IJK SpMSpM time: %lf seconds.\n", overall_time);
        printf("\t- Preparing data before evaluations: %.1lf %%\n", (t2 - t1) / overall_time * 100.0);
        printf("\t- Main IJK loop: %.1lf %%\n", (t3 - t2) / overall_time * 100.0);
        printf("\t- Converting CSR result to Matrix object: %.1lf %%\n", (t4 - t3) / overall_time * 100.0);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
