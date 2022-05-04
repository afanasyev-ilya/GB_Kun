#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename mask_type, typename SemiringT>
void SpMSpM_masked_ikj(const Matrix<mask_type> *_result_mask,
                       const Matrix<T> *_matrix1,
                       const Matrix<T> *_matrix2,
                       Matrix<T> *_matrix_result,
                       SemiringT _op)
{
    cout << "Starting masked ikj algorithm" << endl;
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    auto mask_row_ptr = _result_mask->get_csr()->get_row_ptr();
    auto mask_col_ids_ptr = _result_mask->get_csr()->get_col_ids();

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

    double t2 = omp_get_wtime();

    ENT * matrix1_row_ptr;
    VNT * matrix1_col_ids_ptr;
    T * matrix1_vals_ptr;
    ENT * matrix2_row_ptr;
    VNT * matrix2_col_ids_ptr;
    T * matrix2_vals_ptr;

    MatrixCSR<T> A_csr_first_socket;
    MatrixCSR<T> A_csr_second_socket;
    MatrixCSR<T> B_csr_first_socket;
    MatrixCSR<T> B_csr_second_socket;
    if (num_sockets_used() == 2) {
        cout << "Using NUMA optimization" << endl;
        A_csr_first_socket.deep_copy(_matrix1->get_csr(), 0);
        A_csr_second_socket.deep_copy(_matrix1->get_csr(), 1);
        B_csr_first_socket.deep_copy(_matrix2->get_csr(), 0);
        B_csr_second_socket.deep_copy(_matrix2->get_csr(), 1);
    } else {
        matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
        matrix1_col_ids_ptr = _matrix1->get_csr()->get_col_ids();
        matrix1_vals_ptr = _matrix1->get_csr()->get_vals();
        matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
        matrix2_col_ids_ptr = _matrix2->get_csr()->get_col_ids();
        matrix2_vals_ptr = _matrix2->get_csr()->get_vals();
    }

    #ifdef __USE_KUNPENG__
        const int max_threads_per_socket = sysconf(_SC_NPROCESSORS_ONLN)/2;
    #else
        const int max_threads_per_socket = omp_get_max_threads();
    #endif

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
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
        }

        for (VNT matrix1_row_id = offsets[thread_id].first; matrix1_row_id < offsets[thread_id].second;
             ++matrix1_row_id) {
            unordered_set<ENT> mask_col_ids_set;
            unordered_map<VNT, T> matrix_result;
            for (VNT mask_row_ptr_id = mask_row_ptr[matrix1_row_id];
                 mask_row_ptr_id < mask_row_ptr[matrix1_row_id + 1]; ++mask_row_ptr_id) {
                mask_col_ids_set.insert(mask_col_ids_ptr[mask_row_ptr_id]);
            }

            for (VNT matrix1_row_ptr_id = matrix1_row_ptr[matrix1_row_id];
                 matrix1_row_ptr_id < matrix1_row_ptr[matrix1_row_id + 1]; ++matrix1_row_ptr_id) {
                VNT k = matrix1_col_ids_ptr[matrix1_row_ptr_id];
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

    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);
    double t4 = omp_get_wtime();

    double overall_time = t4 - t1;

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "ikj_masked_mxm", overall_time * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

    printf("Masked IKJ SpMSpM time: %lf seconds.\n", overall_time);
    printf("\t- Preparing data before evaluations: %.1lf %%\n", (t2 - t1) / overall_time * 100.0);
    printf("\t- Main IKJ loop: %.1lf %%\n", (t3 - t2) / overall_time * 100.0);
    printf("\t- Converting CSR result to Matrix object: %.1lf %%\n", (t4 - t3) / overall_time * 100.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
