#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM(const Matrix<T> *_matrix1,
            Matrix<T> *_matrix2,
            Matrix<T> *_matrix_result)
{

    VNT matrix1_num_rows = _matrix1->get_csr()->get_num_rows();
    VNT matrix2_num_cols = _matrix2->get_csc()->get_num_rows();

    vector<VNT> row_ids;
    vector<VNT> col_ids;
    vector<T> values;
    ENT nnz = 0;

    // #pragma omp parallel for
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
                bool matrix2_non_zero = true;
                VNT found_matrix2_row_id;
                VNT left = matrix2_col_start_id;
                VNT right = matrix2_col_end_id - 1;
                while (true) {
                    if (left > right) {
                        matrix2_non_zero = false;
                        break;
                    }
                    VNT mid = left + (right - left) / 2;
                    if (_matrix2->get_csc()->get_col_ids()[mid] < matrix1_col_num) {
                        left = mid + 1;
                    } else if (_matrix2->get_csc()->get_col_ids()[mid] > matrix1_col_num) {
                        right = mid - 1;
                    } else {
                        found_matrix2_row_id = mid;
                        break;
                    }
                }
                if (matrix2_non_zero) {
                    T matrix1_val = _matrix1->get_csr()->get_vals()[matrix1_col_id];
                    T matrix2_val = _matrix2->get_csc()->get_vals()[found_matrix2_row_id];
                    accumulator += matrix1_val * matrix2_val;
                }
            }
            if (accumulator) {
                row_ids.push_back(matrix1_row_id);
                col_ids.push_back(matrix2_col_id);
                values.push_back(accumulator);
                ++nnz;
            }
        }
    }

    SpMSpM_alloc(_matrix_result);
    _matrix_result->build(&row_ids[0], &col_ids[0], &values[0], matrix1_num_rows, nnz);
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
