#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void SpMSpM_unmasked_ikj(const Matrix<T> *_matrix1,
                         const Matrix<T> *_matrix2,
                         Matrix<T> *_matrix_result,
                         SemiringT _op)
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

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto matrix_result = new tsl::hopscotch_map<VNT, T>[n];

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
                    if (matrix_result[i].find(j) == matrix_result[i].end()) {
                        matrix_result[i][j] = identity_val;
                    }
                    matrix_result[i][j] =
                            add_op(matrix_result[i][j],
                                   mul_op(matrix1_val_ptr[matrix1_col_id], matrix2_val_ptr[matrix2_col_id]));
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

    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n);

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

    const auto n = _matrix1->get_csr()->get_num_rows();

    auto matrix_result = new tsl::hopscotch_map<VNT, T>[n];

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
                    if (matrix_result[i].find(j) == matrix_result[i].end()) {
                        matrix_result[i][j] = identity_val;
                    }
                    matrix_result[i][j] =
                            add_op(matrix_result[i][j],
                                   mul_op(matrix1_val_ptr[matrix1_col_id], matrix2_val_ptr[matrix2_col_id]));

                }
            }
        }
    }

    cout << "Applying the mask for ikj algorithm" << endl;
    auto mask_ptr = _result_mask->get_csr();
    auto matrix_result_applied_mask = new tsl::hopscotch_map<VNT, T>[n];
    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            for (VNT col_id = _result_mask->get_csr()->get_row_ptr()[i];
                 col_id < _result_mask->get_csr()->get_row_ptr()[i + 1]; ++col_id) {
                VNT cur_col = _result_mask->get_csr()->get_col_ids()[col_id];
                T cur_val = _result_mask->get_csr()->get_vals()[col_id];
                if (cur_val && matrix_result[i].find(cur_col) != matrix_result[i].end() && matrix_result[i][cur_col]) {
                    matrix_result_applied_mask[i][cur_col] = matrix_result[i][cur_col];
                }
            }
        }
    }

    delete [] matrix_result;
    matrix_result = matrix_result_applied_mask;

    cout << "Counting nnz for ikj algorithm" << endl;
    #pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        for (VNT i = offsets[thread_id].first; i < offsets[thread_id].second; ++i) {
            row_nnz[i] = 0;
            for (auto& [col_id, val] : matrix_result[i]) {
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

    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n);

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

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
