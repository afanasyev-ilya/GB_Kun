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

void balance_matrix_rows(const ENT *_row_ptrs, VNT num_rows, vector<pair<VNT, VNT>> &_offsets)
{
    ENT nnz = _row_ptrs[num_rows];
    int threads_count = omp_get_max_threads();
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

        _offsets.emplace_back(low_val, up_val);
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
    _matrix_result->build(&row_ids[0], &col_ids[0], &values[0], matrix1_num_rows, nnz);
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
    bytes_requested += sizeof(vector<map<VNT, T>>);
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

    vector<unordered_map<VNT, T>> matrix_result(_matrix1->get_csr()->get_num_rows());

    vector<pair<VNT, VNT>> offsets;
    balance_matrix_rows(_matrix1->get_csr()->get_row_ptr(),
                        _matrix1->get_csr()->get_num_rows(),
                        offsets);

    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            for (VNT matrix1_col_id = _matrix1->get_csr()->get_row_ptr()[i];
                 matrix1_col_id < _matrix1->get_csr()->get_row_ptr()[i + 1]; ++matrix1_col_id) {
                VNT k = _matrix1->get_csr()->get_col_ids()[matrix1_col_id];
                for (VNT matrix2_col_id = _matrix2->get_csr()->get_row_ptr()[k];
                     matrix2_col_id < _matrix2->get_csr()->get_row_ptr()[k + 1]; ++matrix2_col_id) {
                    VNT j = _matrix2->get_csr()->get_col_ids()[matrix2_col_id];
                    matrix_result[i][j] += _matrix1->get_csr()->get_vals()[matrix1_col_id] *
                                           _matrix2->get_csr()->get_vals()[matrix2_col_id];
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    SpMSpM_alloc(_matrix_result);
    vector<ENT> row_ptr;
    row_ptr.push_back(0);
    vector<VNT> col_ids;
    vector<T> vals;
    for (VNT i = 0; i < _matrix1->get_csr()->get_num_rows(); ++i) {
        ENT cur_row_size = 0;
        for (const auto& [col_id, val] : matrix_result[i]) {
            if (val != 0) {
                ++cur_row_size;
                col_ids.push_back(col_id);
                vals.push_back(val);
            }
        }
        row_ptr.push_back(row_ptr.back() + cur_row_size);
    }
    // Getting COO format from CSR until needed build functions are implemented:
    vector<VNT> row_ids_coo;
    vector<VNT> col_ids_coo;
    vector<T> values_coo;
    for (VNT i = 0; i < row_ptr.size() - 1; ++i) {
        for (VNT j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            row_ids_coo.push_back(i);
            col_ids_coo.push_back(col_ids[j]);
            values_coo.push_back(vals[j]);
        }
    }
    _matrix_result->build(&row_ids_coo[0], &col_ids_coo[0], &values_coo[0], row_ptr.size() - 1, vals.size());
    double t3 = omp_get_wtime();
    double overall_time = t3 - t1;
    printf("Unmasked IKJ SpMSpM time: %lf seconds.\n", t2-t1);
    printf("Unmasked IKJ SpMSpM converting result time: %lf seconds.\n", t3-t2);
#ifdef __DEBUG_BANDWIDTHS__
    printf("\t- Sustained bandwidth: %lf GB/s\n", bytes_requested / 1e9 / (t2 - t1));
#endif
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
