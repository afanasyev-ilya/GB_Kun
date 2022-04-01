#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void SpMSpM_ijk(const Matrix<T> *_matrix1,
                const Matrix<T> *_matrix2,
                Matrix<T> *_matrix_result,
                const Matrix<T> *_result_mask,
                SemiringT _op)
{
    double t1 = omp_get_wtime();

    _matrix2->sort_csc_rows("STL_SORT");

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT matrix1_num_rows = _matrix1->get_csr()->get_num_rows();
    VNT matrix2_num_cols = _matrix2->get_csc()->get_num_rows();

    vector<VNT> row_ids;
    vector<VNT> col_ids;
    vector<T> values;
    ENT nnz = 0;

    auto offsets = _result_mask->get_csr()->get_load_balancing_offsets();

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT matrix1_row_id = offsets[thread_id].first; matrix1_row_id < offsets[thread_id].second; ++matrix1_row_id) {
            ENT matrix1_col_start_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id];
            ENT matrix1_col_end_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id + 1];
            ENT mask_col_start_id = _result_mask->get_csr()->get_row_ptr()[matrix1_row_id];
            ENT mask_col_end_id = _result_mask->get_csr()->get_row_ptr()[matrix1_row_id + 1];
            for (ENT mask_col_id = mask_col_start_id; mask_col_id < mask_col_end_id; ++mask_col_id) {
                VNT matrix2_col_id = _result_mask->get_csr()->get_col_ids()[mask_col_id];
                VNT matrix2_col_start_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id];
                VNT matrix2_col_end_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id + 1];

                T accumulator = identity_val;
                for (ENT matrix1_col_id = matrix1_col_start_id; matrix1_col_id < matrix1_col_end_id; ++matrix1_col_id) {
                    ENT matrix1_col_num = _matrix1->get_csr()->get_col_ids()[matrix1_col_id];
                    // i == matrix1_row_id, j == matrix2_col_id, k == matrix1_col_num
                    VNT found_matrix2_row_id = spgemm_binary_search(_matrix2->get_csc()->get_col_ids(),
                                                                    matrix2_col_start_id,
                                                                    matrix2_col_end_id - 1,
                                                                    matrix1_col_num);

                    if (found_matrix2_row_id != -1) {
                        T matrix1_val = _matrix1->get_csr()->get_vals()[matrix1_col_id];
                        T matrix2_val = _matrix2->get_csc()->get_vals()[found_matrix2_row_id];
                        accumulator = add_op(accumulator, mul_op(matrix1_val, matrix2_val));
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

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
