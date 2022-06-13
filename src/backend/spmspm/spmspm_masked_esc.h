#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "esc_helpers.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename mask_type, typename SemiringT>
void SpMSpM_masked_esc(const Matrix<mask_type> *_result_mask,
                       const Matrix<T> *_matrix1,
                       const Matrix<T> *_matrix2,
                       Matrix<T> *_matrix_result,
                       SemiringT _op)
{
    static const unsigned long long cache_effective_size = 1048675;
    const unsigned long long nnz_cache = 1048675;
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto row_nnz = new ENT[n];
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        row_nnz[i] = 0;
    }

    auto offsets = _matrix1->get_csr()->get_load_balancing_offsets();

    size_t threads_num = offsets.size();

    vector<vector<VNT> > thread_slice_cols(threads_num, vector<VNT>());
    vector<vector<T> > thread_slice_vals(threads_num, vector<T>());

    auto matrix1_val_ptr = _matrix1->get_csr()->get_vals();
    auto matrix2_val_ptr = _matrix2->get_csr()->get_vals();
    auto matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
    auto matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
    auto matrix1_col_ptr = _matrix1->get_csr()->get_col_ids();
    auto matrix2_col_ptr = _matrix2->get_csr()->get_col_ids();
    auto mask_row_ptr = _result_mask->get_csr()->get_row_ptr();
    auto mask_col_ptr = _result_mask->get_csr()->get_col_ids();
    auto mask_val_ptr = _result_mask->get_csr()->get_vals();

    double t2 = omp_get_wtime();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        auto cur_slice_start = offsets[thread_id].first;
        auto cur_slice_end = offsets[thread_id].second;
        if (cur_slice_start > cur_slice_end) {
            cur_slice_end = cur_slice_start;
        }
        if (cur_slice_start > n) {
            cur_slice_start = n;
        }
        if (cur_slice_end > n) {
            cur_slice_end = n;
        }

        // setting block size by the dynamic parameter tuning method
        int blocks_per_slice;
        unsigned long long block_size;
        const auto cur_slice_length = cur_slice_end - cur_slice_start;
        if (cur_slice_length) {
            ENT cur_slice_nnz =
                matrix1_row_ptr[cur_slice_end] - matrix1_row_ptr[cur_slice_start];
            unsigned long long avg_nnz = (cur_slice_nnz + cur_slice_length - 1) / cur_slice_length;
            const auto rounded_avg_nnz = esc_helpers::get_nearest_power_of_two(avg_nnz);

            block_size = nnz_cache / rounded_avg_nnz;
            blocks_per_slice = (cur_slice_length + block_size - 1) / block_size;
        } else {
            blocks_per_slice = 0;
            block_size = 0;
        }

        // doing ESC algorithm for each block
        for (int cur_block_id = 0; cur_block_id < blocks_per_slice; ++cur_block_id) {
            auto cur_block_start = cur_slice_start + cur_block_id * block_size;
            auto cur_block_end = cur_slice_start + (cur_block_id + 1) * block_size;
            if (cur_block_start > cur_slice_end) {
                cur_block_start = cur_slice_end;
            }
            if (cur_block_end > cur_slice_end) {
                cur_block_end = cur_slice_end;
            }

            vector<tuple<VNT, ENT, T> > cur_accumulator;

            // expansion
            for (VNT i = cur_block_start; i < cur_block_end; ++i) {
                unordered_set<ENT> mask_col_ids_set;
                for (VNT mask_row_ptr_id = mask_row_ptr[i];
                     mask_row_ptr_id < mask_row_ptr[i + 1]; ++mask_row_ptr_id) {
                    if (mask_val_ptr[mask_row_ptr_id]) {
                        mask_col_ids_set.insert(mask_col_ptr[mask_row_ptr_id]);
                    }
                }

                for (VNT matrix1_col_id = matrix1_row_ptr[i];
                     matrix1_col_id < matrix1_row_ptr[i + 1]; ++matrix1_col_id) {
                    VNT k = matrix1_col_ptr[matrix1_col_id];
                    for (VNT matrix2_col_id = matrix2_row_ptr[k];
                         matrix2_col_id < matrix2_row_ptr[k + 1]; ++matrix2_col_id) {
                        ENT j = matrix2_col_ptr[matrix2_col_id];
                        if (mask_col_ids_set.find(j) == mask_col_ids_set.end()) {
                            continue;
                        }
                        T cur_val = mul_op(matrix1_val_ptr[matrix1_col_id],
                                           matrix2_val_ptr[matrix2_col_id]);
                        if (cur_val) {
                            cur_accumulator.push_back(make_tuple(i, j, cur_val));
                        }
                    }
                }
            }
            // sorting
            std::sort(cur_accumulator.begin(), cur_accumulator.end());
            // compression
            VNT prev_row_id;
            VNT prev_col_id;
            if (!cur_accumulator.empty()) {
                prev_row_id = std::get<0>(cur_accumulator[0]);
                prev_col_id = std::get<1>(cur_accumulator[0]);
            }
            T cur_val = identity_val;
            ENT cur_row_nnz = 0;
            for (const auto &el : cur_accumulator) {
                if (std::get<0>(el) != prev_row_id || std::get<1>(el) != prev_col_id) {
                    if (cur_val) {
                        thread_slice_cols[thread_id].push_back(prev_col_id);
                        thread_slice_vals[thread_id].push_back(cur_val);
                        ++cur_row_nnz;
                    }
                    if (std::get<0>(el) != prev_row_id) {
                        row_nnz[prev_row_id] = cur_row_nnz;
                        cur_row_nnz = 0;
                    }
                    prev_row_id = std::get<0>(el);
                    prev_col_id = std::get<1>(el);
                    cur_val = add_op(identity_val, std::get<2>(el));
                } else {
                    cur_val = add_op(cur_val, std::get<2>(el));
                }
            }
            if (!cur_accumulator.empty() && cur_val) {
                thread_slice_cols[thread_id].push_back(prev_col_id);
                thread_slice_vals[thread_id].push_back(cur_val);
                ++cur_row_nnz;
            }
            if (!cur_accumulator.empty()) {
                row_nnz[prev_row_id] = cur_row_nnz;
            }
        }
    }

    double t3 = omp_get_wtime();

    ENT nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (VNT i = 0; i < n; ++i) {
        nnz += row_nnz[i];
    }

    auto row_ptr = new ENT[n + 1];
    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n);

    auto col_ids = new VNT[nnz];
    VNT col_ids_offset = 0;
    for (const auto &thread_col_vector : thread_slice_cols) {
        for (const auto &col_id : thread_col_vector) {
            col_ids[col_ids_offset] = col_id;
            ++col_ids_offset;
        }
    }

    auto vals = new T[nnz];
    VNT vals_offset = 0;
    for (const auto &thread_val_vector : thread_slice_vals) {
        for (const auto &val : thread_val_vector) {
            vals[vals_offset] = val;
            ++vals_offset;
        }
    }

    double t4 = omp_get_wtime();
    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);

    double t5 = omp_get_wtime();

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_masked_mxm_total_time", (t5 - t1) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_masked_mxm_inner_loop", (t3 - t2) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_masked_mxm_preparation", (t2 - t1) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_masked_mxm_csr_export", (t5 - t3) * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

