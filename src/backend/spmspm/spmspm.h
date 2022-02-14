#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM(const Matrix<T> *_matrix1,
            const Matrix<T> *_matrix2,
            Matrix<T> *_matrix_result,
            const string& mode)
{
    double t1 = omp_get_wtime();
    VNT matrix1_num_rows = _matrix1->get_csr()->get_num_rows();
    VNT matrix2_num_cols = _matrix2->get_csc()->get_num_rows();

    vector<VNT> row_ids;
    vector<VNT> col_ids;
    vector<T> values;
    ENT nnz = 0;

    // #pragma omp parallel for default(none), shared(_matrix1, _matrix2, _matrix_result, matrix1_num_rows, matrix2_num_cols, row_ids, col_ids, values, nnz)
    for (VNT matrix1_row_id = 0; matrix1_row_id < matrix1_num_rows; ++matrix1_row_id) {
        ENT matrix1_col_start_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id];
        ENT matrix1_col_end_id = _matrix1->get_csr()->get_row_ptr()[matrix1_row_id + 1];
        for (VNT matrix2_col_id = 0; matrix2_col_id < matrix2_num_cols; ++matrix2_col_id) {
            VNT matrix2_col_start_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id];
            VNT matrix2_col_end_id = _matrix2->get_csc()->get_row_ptr()[matrix2_col_id + 1];

            T accumulator = 0;
            for (ENT matrix1_col_id = matrix1_col_start_id; matrix1_col_id < matrix1_col_end_id; ++matrix1_col_id) {
                if (matrix1_col_id != matrix1_col_start_id &&
                        _matrix1->get_csr()->get_col_ids()[matrix1_col_id] ==
                        _matrix1->get_csr()->get_col_ids()[matrix1_col_id - 1]) {
                    continue;
                }
                ENT matrix1_col_num = _matrix1->get_csr()->get_col_ids()[matrix1_col_id];
                // i == matrix1_row_id, j == matrix2_col_id, k == matrix1_col_num
                // Вынести в отдельную функцию
                bool matrix2_non_zero = false;
                VNT found_matrix2_row_id;
                VNT left = matrix2_col_start_id;
                VNT right = matrix2_col_end_id - 1;
                while (true) {
                    if (left > right) {
                        break;
                    }
                    VNT mid = left + (right - left) / 2;
                    if (_matrix2->get_csc()->get_col_ids()[mid] < matrix1_col_num) {
                        left = mid + 1;
                    } else if (_matrix2->get_csc()->get_col_ids()[mid] > matrix1_col_num) {
                        right = mid - 1;
                    } else {
                        found_matrix2_row_id = mid;
                        matrix2_non_zero = true;
                        break;
                    }
                }
                if (matrix2_non_zero) {
                    T matrix1_val = _matrix1->get_csr()->get_vals()[matrix1_col_id];
                    T matrix2_val = _matrix2->get_csc()->get_vals()[found_matrix2_row_id];
                    accumulator += matrix1_val * matrix2_val;
                }
            }
            // #pragma omp critical(updateresults)
            if (accumulator) {
                row_ids.push_back(matrix1_row_id);
                col_ids.push_back(matrix2_col_id);
                values.push_back(accumulator);
                ++nnz;
            }
        }
    }
    double t2 = omp_get_wtime();
    SpMSpM_alloc(_matrix_result);
    _matrix_result->build(&row_ids[0], &col_ids[0], &values[0], matrix1_num_rows, nnz);
    double t3 = omp_get_wtime();

    int error_cnt = 0;
    for (int i = 0; i < matrix1_num_rows; ++i) {
        for (int j = 0; j < matrix1_num_rows; ++j) {
            T accumulator = 0;
            for (int k = 0; k < matrix1_num_rows; ++k) {
                accumulator += _matrix1->get_csr()->get(i, k) * _matrix2->get_csr()->get(k, j);
            }
            if (_matrix_result->get_csr()->get(i, j) != accumulator) {
                std::cout << i << ' ' << j << " " << accumulator << " " << _matrix_result->get_csr()->get(i, j) << std::endl;
                ++error_cnt;
            }
        }
    }
    double overall_time = t3 - t1;
    printf("SpMSpM time: %lf seconds.\n", t3-t1);
    printf("\t- Calculating result: %.1lf %%\n", (t2 - t1) / overall_time * 100.0);
    printf("\t- Converting result: %.1lf %%\n", (t3 - t2) / overall_time * 100.0);
    printf("\t- Error cnt: %d\n", error_cnt);
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
