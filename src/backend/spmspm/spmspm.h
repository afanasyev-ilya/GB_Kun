#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VNT binary_search(const Index* data, VNT left, VNT right, ENT value)
{
    while (true) {
        if (left > right) {
            return -1;
        }
        VNT mid = left + (right - left) / 2; // div 2 or div rand
        if (data[mid] < value) {
            left = mid + 1;
        } else if (data[mid] > value) {
            right = mid - 1;
        } else {
            return mid;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void balance_matrix_rows(const ENT *_row_ptrs, VNT num_rows, pair<VNT, VNT> *_offsets, int threads_count)
{
    ENT nnz = _row_ptrs[num_rows];
    ENT approx_nnz_per_thread = (nnz - 1) / threads_count + 1;
    for(int tid = 0; tid < threads_count; tid++)
    {
        ENT expected_tid_left_border = approx_nnz_per_thread * tid;
        ENT expected_tid_right_border = approx_nnz_per_thread * (tid + 1);

        const ENT* left_border_ptr = _row_ptrs;
        const ENT* right_border_ptr = _row_ptrs + num_rows;

        auto low_pos = std::lower_bound(left_border_ptr, right_border_ptr, expected_tid_left_border);
        auto up_pos = std::lower_bound(left_border_ptr, right_border_ptr, expected_tid_right_border);

        VNT low_val = low_pos - left_border_ptr;
        VNT up_val = min(num_rows, (VNT) (up_pos - left_border_ptr));

        _offsets[tid] = make_pair(low_val, up_val);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM_unmasked_ijk(const Matrix<T> *_matrix1,
                     const Matrix<T> *_matrix2,
                     Matrix<T> *_matrix_result)
{
    double t1 = omp_get_wtime();
    VNT matrix1_num_rows = _matrix1->get_csr()->get_num_rows();
    VNT matrix2_num_cols = _matrix2->get_csc()->get_num_rows();

    vector<VNT> row_ids;
    vector<VNT> col_ids;
    vector<T> values;
    ENT nnz = 0;

    #pragma omp parallel for default(none), shared(_matrix1, _matrix2, _matrix_result, matrix1_num_rows, matrix2_num_cols, row_ids, col_ids, values, nnz)
    for (VNT matrix1_row_id = 0; matrix1_row_id < matrix1_num_rows; ++matrix1_row_id) {
        ENT matrix1_col_start_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id];
        ENT matrix1_col_end_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id + 1];
        for (VNT matrix2_col_id = 0; matrix2_col_id < matrix2_num_cols; ++matrix2_col_id) {
            VNT matrix2_col_start_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id];
            VNT matrix2_col_end_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id + 1];

            T accumulator = 0;
            for (ENT matrix1_col_id = matrix1_col_start_id; matrix1_col_id < matrix1_col_end_id; ++matrix1_col_id) {
                ENT matrix1_col_num = _matrix1->get_csr()->get_col_ids()[matrix1_col_id];
                // i == matrix1_row_id, j == matrix2_col_id, k == matrix1_col_num
                VNT found_matrix2_row_id = binary_search(_matrix2->get_csc()->get_col_ids(),
                                                         matrix2_col_start_id,
                                                         matrix2_col_end_id - 1,
                                                         matrix1_col_num);

                if (found_matrix2_row_id != -1) {
                    T matrix1_val = _matrix1->get_csr()->get_vals()[matrix1_col_id];
                    T matrix2_val = _matrix2->get_csc()->get_vals()[found_matrix2_row_id];
                    accumulator += matrix1_val * matrix2_val;
                }
            }
            #pragma omp critical(updateresults)
            {
                if (accumulator) {
                    row_ids.push_back(matrix1_row_id);
                    col_ids.push_back(matrix2_col_id);
                    values.push_back(accumulator);
                    ++nnz;
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    SpMSpM_alloc(_matrix_result);
    _matrix_result->build(&row_ids[0], &col_ids[0], &values[0],  nnz);
    double t3 = omp_get_wtime();
    double overall_time = t3 - t1;
    printf("Unmasked IJK SpMSpM time: %lf seconds.\n", t3-t1);
    printf("\t- Calculating result: %.1lf %%\n", (t2 - t1) / overall_time * 100.0);
    printf("\t- Converting result: %.1lf %%\n", (t3 - t2) / overall_time * 100.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM_unmasked_ikj(const Matrix<T> *_matrix1,
                         const Matrix<T> *_matrix2,
                         Matrix<T> *_matrix_result)
{

#ifdef __DEBUG_BANDWIDTHS__
    double bytes_requested = 0;
    bytes_requested += sizeof(vector<unordered_map<VNT, T>>);
    #pragma omp parallel for reduction(+:bytes_requested)
    for (VNT i = 0; i < _matrix1->get_csr()->get_num_rows(); ++i) {
        bytes_requested += sizeof(_matrix1->get_csr()->get_num_rows());
        for (VNT matrix1_col_id = _matrix1->get_csr()->get_row_ptr()[i];
             matrix1_col_id < _matrix1->get_csr()->get_row_ptr()[i + 1]; ++matrix1_col_id) {
            bytes_requested += sizeof(_matrix1->get_csr()->get_row_ptr()[i]);
            bytes_requested += sizeof(_matrix1->get_csr()->get_row_ptr()[i + 1]);
            VNT k = _matrix1->get_csr()->get_col_ids()[matrix1_col_id];
            bytes_requested += sizeof(_matrix1->get_csr()->get_col_ids()[matrix1_col_id]);
            for (VNT matrix2_col_id = _matrix2->get_csr()->get_row_ptr()[k];
                 matrix2_col_id < _matrix2->get_csr()->get_row_ptr()[k + 1]; ++matrix2_col_id) {
                bytes_requested += sizeof(_matrix2->get_csr()->get_row_ptr()[k]);
                bytes_requested += sizeof(_matrix2->get_csr()->get_row_ptr()[k + 1]);
                VNT j = _matrix2->get_csr()->get_col_ids()[matrix2_col_id];
                bytes_requested += sizeof(_matrix2->get_csr()->get_col_ids()[matrix2_col_id]);
                bytes_requested += sizeof(_matrix1->get_csr()->get_vals()[matrix1_col_id]);
                bytes_requested += sizeof(_matrix2->get_csr()->get_vals()[matrix2_col_id]);
                bytes_requested += sizeof(T);
            }
        }
    }
#endif
    double t1 = omp_get_wtime();

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto matrix_result = new unordered_map<VNT, T>[n];

    auto row_nnz = new ENT[n];

    int threads_count = omp_get_max_threads();
    auto offsets = _matrix1->get_csr()->get_load_balancing_offsets();


    auto matrix1_val_ptr = _matrix1->get_csr()->get_vals();
    auto matrix2_val_ptr = _matrix2->get_csr()->get_vals();
    auto matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
    auto matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
    auto matrix1_col_ptr = _matrix1->get_csr()->get_col_ids();
    auto matrix2_col_ptr = _matrix2->get_csr()->get_col_ids();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            for (VNT matrix1_col_id = matrix1_row_ptr[i]; matrix1_col_id < matrix1_row_ptr[i + 1]; ++matrix1_col_id) {
                VNT k = matrix1_col_ptr[matrix1_col_id];
                for (VNT matrix2_col_id = matrix2_row_ptr[k]; matrix2_col_id < matrix2_row_ptr[k + 1]; ++matrix2_col_id) {
                    VNT j = matrix2_col_ptr[matrix2_col_id];
                    matrix_result[i][j] += matrix1_val_ptr[matrix1_col_id] * matrix2_val_ptr[matrix2_col_id];
                }
            }
            row_nnz[i] = 0;
            for (const auto& [col_id, val] : matrix_result[i]) {
                if (val != 0) {
                    ++row_nnz[i];
                }
            }
        }
    }

    double t2 = omp_get_wtime();

    ENT nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (VNT i = 0; i < n; ++i) {
        nnz += row_nnz[i];
    }

    auto row_ptr = new ENT[n + 1];

    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n, row_ptr, 0);

    auto col_ids = new VNT[nnz];
    auto vals = new T[nnz];

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            ENT cur_col_ids_start = row_ptr[i];
            ENT cur_col_ids_offset = 0;
            for (const auto& [col_id, val] : matrix_result[i]) {
                if (val != 0) {
                    col_ids[cur_col_ids_start + cur_col_ids_offset] = col_id;
                    vals[cur_col_ids_start + cur_col_ids_offset] = val;
                    ++cur_col_ids_offset;
                }
            }
        }
    }

    delete [] matrix_result;
    // delete [] offsets;

    double t3 = omp_get_wtime();

    double my_bw = 0;
    #ifdef __DEBUG_BANDWIDTHS__
    my_bw = bytes_requested / 1e9 / (t2 - t1);
    #endif

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "Hash_Based_mxm", (t3 - t1) * 1000, 0.0, my_bw, 0ll);
    fclose(my_f);

    double t4 = omp_get_wtime();

    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);

    double t5 = omp_get_wtime();

    printf("Unmasked IKJ SpMSpM main loop: %lf seconds.\n", t2-t1);
    printf("Unmasked IKJ SpMSpM converting result hash-map to CSR time: %lf seconds.\n", t3-t2);
    printf("Unmasked IKJ SpMSpM exporting results to a file time: %lf seconds.\n", t4-t3);
    printf("Unmasked IKJ SpMSpM converting CSR to Matrix object time: %lf seconds.\n", t5-t4);
    printf("Unmasked IKJ SpMSpM total time: %lf seconds.\n", t5-t1);
#ifdef __DEBUG_BANDWIDTHS__
    printf("\t- Sustained bandwidth: %lf GB/s\n", bytes_requested / 1e9 / (t2 - t1));
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename mask_type>
void SpMSpM_masked_ikj(const Matrix<mask_type> *_result_mask,
                       const Matrix<T> *_matrix1,
                       const Matrix<T> *_matrix2,
                       Matrix<T> *_matrix_result)
{
    double t1 = omp_get_wtime();

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto matrix_result = new unordered_map<VNT, T>[n];

    auto row_nnz = new ENT[n];

    int threads_count = omp_get_max_threads();
    auto offsets = _matrix1->get_csr()->get_load_balancing_offsets();

    auto matrix1_val_ptr = _matrix1->get_csr()->get_vals();
    auto matrix2_val_ptr = _matrix2->get_csr()->get_vals();
    auto matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
    auto matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
    auto matrix1_col_ptr = _matrix1->get_csr()->get_col_ids();
    auto matrix2_col_ptr = _matrix2->get_csr()->get_col_ids();
    auto mask_ptr = _result_mask->get_csr();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            for (VNT matrix1_col_id = matrix1_row_ptr[i]; matrix1_col_id < matrix1_row_ptr[i + 1]; ++matrix1_col_id) {
                VNT k = matrix1_col_ptr[matrix1_col_id];
                for (VNT matrix2_col_id = matrix2_row_ptr[k]; matrix2_col_id < matrix2_row_ptr[k + 1]; ++matrix2_col_id) {
                    VNT j = matrix2_col_ptr[matrix2_col_id];
                    if (mask_ptr->get(i, j)) {
                        matrix_result[i][j] +=
                                matrix1_val_ptr[matrix1_col_id] * matrix2_val_ptr[matrix2_col_id];
                    }
                }
            }
            row_nnz[i] = 0;
            for (const auto& [col_id, val] : matrix_result[i]) {
                if (val != 0) {
                    ++row_nnz[i];
                }
            }
        }
    }

    double t2 = omp_get_wtime();

    ENT nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (VNT i = 0; i < n; ++i) {
        nnz += row_nnz[i];
    }

    auto row_ptr = new ENT[n + 1];

    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n, row_ptr, 0);

    auto col_ids = new VNT[nnz];
    auto vals = new T[nnz];

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            ENT cur_col_ids_start = row_ptr[i];
            ENT cur_col_ids_offset = 0;
            for (const auto& [col_id, val] : matrix_result[i]) {
                if (val != 0) {
                    col_ids[cur_col_ids_start + cur_col_ids_offset] = col_id;
                    vals[cur_col_ids_start + cur_col_ids_offset] = val;
                    ++cur_col_ids_offset;
                }
            }
        }
    }

    delete [] matrix_result;
    // delete [] offsets;

    double t3 = omp_get_wtime();

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "Hash_Based_masked_mxm", (t3 - t1) * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

    double t4 = omp_get_wtime();

    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);

    double t5 = omp_get_wtime();

    printf("Masked IKJ SpMSpM main loop: %lf seconds.\n", t2-t1);
    printf("Masked IKJ SpMSpM converting result hash-map to CSR time: %lf seconds.\n", t3-t2);
    printf("Masked IKJ SpMSpM exporting results to a file time: %lf seconds.\n", t4-t3);
    printf("Masked IKJ SpMSpM converting CSR to Matrix object time: %lf seconds.\n", t5-t4);
    printf("Masked IKJ SpMSpM total time: %lf seconds.\n", t5-t1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM_alloc(Matrix<T> *_matrix_result)
{
    _matrix_result->set_preferred_matrix_format(CSR);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
