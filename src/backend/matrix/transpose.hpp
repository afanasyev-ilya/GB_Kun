#ifndef GB_KUN_TRANSPOSE_HPP
#define GB_KUN_TRANSPOSE_HPP

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
LA_Info Matrix<T>::transpose()
{
    #ifdef __PARALLEL_TRANSPOSE__
    transpose_parallel();
    #elif
    transpose_sequential();
    #endif
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Matrix<T>::transpose_sequential()
{

    memset(csc_data->get_row_ptr(),0, (csc_data->get_num_rows() + 1) * sizeof(Index));
    memset(csc_data->get_col_ids(),0, csc_data->get_nnz()* sizeof(Index));
    memset(csc_data->get_vals(),0, csc_data->get_nnz()* sizeof(T));

    VNT csr_ncols = csr_data->get_num_cols();
    VNT csr_nrows = csr_data->get_num_rows();
    auto curr = new int[csr_ncols]();

    for (Index i = 0; i < csr_nrows; i++){
        for (Index j = csr_data->get_row_ptr()[i]; j < csr_data->get_row_ptr()[i+1]; j++) {
            csc_data->get_row_ptr()[csr_data->get_col_ids()[j] + 1]++;
        }
    }
    for (Index i = 1; i < csr_ncols + 1; i++){
        csc_data->get_row_ptr()[i] += csc_data->get_row_ptr()[i - 1];
    }
    for (Index i = 0; i < csr_nrows; i++){
        for (Index j = csr_data->get_row_ptr()[i]; j < csr_data->get_row_ptr()[i+1]; j++) {
            auto loc = csc_data->get_row_ptr()[csr_data->get_col_ids()[j]] + curr[csr_data->get_col_ids()[j]]++;
            csc_data->get_col_ids()[loc] = i;
            csc_data->get_vals()[loc] = csr_data->get_vals()[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void Matrix<T>::transpose_parallel(void) {
    double mem_a = omp_get_wtime();
    memset(csc_data->get_row_ptr(),0, (csc_data->get_num_rows() + 1) * sizeof(Index));
    memset(csc_data->get_col_ids(),0, csc_data->get_nnz()* sizeof(Index));
    memset(csc_data->get_vals(),0, csc_data->get_nnz()* sizeof(T));

    VNT csr_ncols = csr_data->get_num_cols();
    VNT csr_nrows = csr_data->get_num_rows();
    auto dloc = new Index[csr_data->get_nnz()]();
    double mem_b = omp_get_wtime();

    Index temp;
    Index* row_ptr = csc_data->get_row_ptr();

    double fetch_a = omp_get_wtime();
    auto offsets = this->get_csr()->get_load_balancing_offsets();

    #pragma omp parallel shared(csr_nrows, csr_ncols, row_ptr, dloc)
    {
    int tid = omp_get_thread_num();
    VNT first_row = offsets[tid].first;
    VNT last_row = offsets[tid].second;

    for(VNT row = first_row; row < last_row; row++) {
        for (int j = csr_data->get_row_ptr()[row]; j < csr_data->get_row_ptr()[row + 1]; j++) {
            dloc[j] = my_fetch_add(&row_ptr[csr_data->get_col_ids()[j]], static_cast<Index>(1));
        }
    }
    }

    double fetch_b = omp_get_wtime();

    double scan_a = omp_get_wtime();

    ParallelPrimitives::exclusive_scan(csc_data->get_row_ptr(),csc_data->get_row_ptr(),csr_ncols, csc_data->get_row_ptr(), 0);
    double scan_b = omp_get_wtime();

    double final_a = omp_get_wtime();
    #pragma omp parallel shared(csr_nrows, csr_ncols, row_ptr, dloc)
    {
    int tid = omp_get_thread_num();
    VNT first_row = offsets[tid].first;
    VNT last_row = offsets[tid].second;
    for(VNT row = first_row; row < last_row; row++) {
        for (Index j = csr_data->get_row_ptr()[row]; j < csr_data->get_row_ptr()[row + 1]; j++) {
            auto loc = csc_data->get_row_ptr()[csr_data->get_col_ids()[j]] + dloc[j];
            csc_data->get_col_ids()[loc] = row;
            csc_data->get_vals()[loc] = csr_data->get_vals()[j];
        }
    }
    }
    #pragma omp barrier
    double final_b = omp_get_wtime();

    if(num_sockets_used() > 1)
    {
        csc_data->numa_aware_realloc();
    }

    #ifdef __DEBUG_BANDWIDTHS__
    std::cout << "Inner time for mem " << mem_b - mem_a << " seconds" << std::endl;
    std::cout << "Inner time for fetch " << fetch_b - fetch_a << " seconds" << std::endl;
    std::cout << "Inner time for scan " << scan_b - scan_a << " seconds" << std::endl;
    std::cout << "Inner time for final " << final_b - final_a << " seconds" << std::endl;

    double total_bw  =
            2 * sizeof(Index) * csr_data->get_num_rows() +
            2 * sizeof(Index) * csr_data->get_nnz() +
            sizeof(Index) * csr_data->get_num_rows() +
            2 * sizeof(Index) * csr_data->get_nnz() +
            2 * sizeof(Index) * omp_get_num_threads() +
            2 * sizeof(Index) * csr_data->get_nnz() +
            2 * sizeof(Index) * csr_data->get_num_rows() +
            2 * sizeof(Index) * csr_data->get_nnz() +
            sizeof(Index) * csr_data->get_num_rows() +
            1 * sizeof(Index) * csr_data->get_nnz() +
            2 * sizeof(T) * csr_data->get_nnz() +

                    (csc_data->get_num_rows() + 1) * sizeof(Index) + csc_data->get_nnz()* sizeof(Index) +
                    csc_data->get_nnz()* sizeof(T);

    total_bw /= (final_b - mem_a);

    std::cout << "Overall bandwidth " << total_bw / 1000000000 << "GByte/sec" << std::endl;
    #endif
}

#endif //GB_KUN_TRANSPOSE_HPP
