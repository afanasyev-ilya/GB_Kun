#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void SpMSpM_unmasked_esc(const Matrix<T> *_matrix1,
                         const Matrix<T> *_matrix2,
                         Matrix<T> *_matrix_result,
                         SemiringT _op)
{
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto offsets = _matrix1->get_csr()->get_load_balancing_offsets();

    auto matrix1_val_ptr = _matrix1->get_csr()->get_vals();
    auto matrix2_val_ptr = _matrix2->get_csr()->get_vals();
    auto matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
    auto matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
    auto matrix1_col_ptr = _matrix1->get_csr()->get_col_ids();
    auto matrix2_col_ptr = _matrix2->get_csr()->get_col_ids();

    double t2 = omp_get_wtime();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        const auto cur_slice_start = offsets[thread_id].first;
        const auto cur_slice_end = offsets[thread_id].second;
        const auto cur_slice_length = cur_slice_end - cur_slice_start;
        // setting block size by the dynamic parameter tuning method
        const int block_size = 5; // to be implemented
        const int blocks_per_slice = (cur_slice_length + block_size - 1) / block_size;
        for (int cur_block_id = 0; cur_block_id < blocks_per_slice; ++cur_block_id) {
            const auto cur_block_start = cur_slice_start + cur_block_id * block_size;
            const auto cur_block_end = cur_slice_start + cur_block_id * block_size + 1;
            if (cur_slice_end > n) {
                cur_block_end = n;
            }
            vector<tuple<VNT, ENT, T> > cur_accumulator;
            // expansion
            for (VNT i = cur_block_start; i < cur_block_end; ++i) {
                for (VNT matrix1_col_id = matrix1_row_ptr[i];
                     matrix1_col_id < matrix1_row_ptr[i + 1]; ++matrix1_col_id) {
                    VNT k = matrix1_col_ptr[matrix1_col_id];
                    for (VNT matrix2_col_id = matrix2_row_ptr[k];
                         matrix2_col_id < matrix2_row_ptr[k + 1]; ++matrix2_col_id) {
                        ENT j = matrix2_col_ptr[matrix2_col_id];
                        cur_accumulator.push_back(make_tuple(i, j,
                                                             mul_op(matrix1_val_ptr[matrix1_col_id],
                                                                    matrix2_val_ptr[matrix2_col_id])));
                    }
                }
            }
            // sorting
            std::sort(cur_accumulator.begin(), cur_accumulator.end());
            // compression

            // implement addition of the identity_val
        }
    }

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_mxm", (t2 - t1) * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
