/// @file spmspm_hash_based_gustavson.h
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Unmasked hash-based IKJ SpMSpM algorithm
/// @details Implements Unmasked hash-based IKJ SpMSpM algorithm
/// @date June 8, 2022

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @namespace Lablas
namespace lablas {

/// @namespace Backend
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Hash-based unmasked SpMSpM algorithm.
///
/// This algorithm by default uses tsl::hopscotch_map hash map container to accumulate matrix elements.
/// @param[in] _matrix1 Pointer to the first input matrix
/// @param[in] _matrix2 Pointer to the second input matrix
/// @param[out] _matrix_result Pointer to the (empty) matrix object that will contain the result matrix
/// @param[in] _op Semiring operation
template <typename T, typename SemiringT>
void SpMSpM_unmasked_ikj(const Matrix<T> *_matrix1,
                         const Matrix<T> *_matrix2,
                         Matrix<T> *_matrix_result,
                         SemiringT _op)
{
    LOG_TRACE("Running SpMSpM_unmasked_ikj")
    // Same IKJ loop to measure bandwidth
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

    // Each row will have its own copy of hash map
    auto matrix_result = new tsl::hopscotch_map<VNT, T>[n];

    auto row_nnz = new ENT[n];

    int threads_count = omp_get_max_threads();
    auto offsets = _matrix1->get_csr()->get_load_balancing_offsets();

    // explicit optimization for saving pointer's values on stack variables
    auto matrix1_val_ptr = _matrix1->get_csr()->get_vals();
    auto matrix2_val_ptr = _matrix2->get_csr()->get_vals();
    auto matrix1_row_ptr = _matrix1->get_csr()->get_row_ptr();
    auto matrix2_row_ptr = _matrix2->get_csr()->get_row_ptr();
    auto matrix1_col_ptr = _matrix1->get_csr()->get_col_ids();
    auto matrix2_col_ptr = _matrix2->get_csr()->get_col_ids();

    // main IKJ loop
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
                    // updating result in hash map
                    matrix_result[i][j] =
                            add_op(matrix_result[i][j],
                                   mul_op(matrix1_val_ptr[matrix1_col_id], matrix2_val_ptr[matrix2_col_id]));
                }
            }
            // after finishing i-th row updating nnz count for this row
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
    // building CSR row_ptr array based on each row nnz count
    ParallelPrimitives::exclusive_scan(row_nnz, row_ptr, n);

    auto col_ids = new VNT[nnz];
    auto vals = new T[nnz];

    // building col ids and vals from hash maps avoiding zero values
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

    // saving performance results into the output file
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "Hash_Based_mxm", (t3 - t1) * 1000, 0.0, my_bw, 0ll);
    fclose(my_f);

    double t4 = omp_get_wtime();

    // Building result matrix from CSR arrays
    SpMSpM_alloc(_matrix_result);
    _matrix_result->build_from_csr_arrays(row_ptr, col_ids, vals, n, nnz);

    double t5 = omp_get_wtime();
    #ifdef __DEBUG_INFO__
        printf("Unmasked IKJ SpMSpM main loop: %lf seconds.\n", t2-t1);
        printf("Unmasked IKJ SpMSpM converting result hash-map to CSR time: %lf seconds.\n", t3-t2);
        printf("Unmasked IKJ SpMSpM exporting results to a file time: %lf seconds.\n", t4-t3);
        printf("Unmasked IKJ SpMSpM converting CSR to Matrix object time: %lf seconds.\n", t5-t4);
        printf("Unmasked IKJ SpMSpM total time: %lf seconds.\n", t5-t1);
    #endif
    #ifdef __DEBUG_BANDWIDTHS__
        printf("\t- Sustained bandwidth: %lf GB/s\n", bytes_requested / 1e9 / (t2 - t1));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
