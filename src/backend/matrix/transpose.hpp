#ifndef GB_KUN_TRANSPOSE_HPP
#define GB_KUN_TRANSPOSE_HPP

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Index detect_row_idx(ENT *row_ptr, Index idx, VNT nrows, Index start_index) {
    for (Index i = start_index; i < nrows; i++) {
        if (idx >= row_ptr[i] && idx < row_ptr[i+1]) {
            return i;
        }
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
LA_Info Matrix<T>::transpose()
{
    ptr_swap(csr_data, csc_data);
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
LA_Info Matrix<T>::csr_to_csc()
{
    #ifdef __PARALLEL_TRANSPOSE__
    transpose_parallel();
    #else
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
    for (Index i = 1; i < csr_ncols + 1; i++) {
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
    int max_threads = omp_get_max_threads();
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

    ParallelPrimitives::exclusive_scan(csc_data->get_row_ptr(),csc_data->get_row_ptr(),csr_ncols);
    double scan_b = omp_get_wtime();

    CommonWorkspace ccp(csr_data->get_num_cols() + 1, csc_data->get_row_ptr());

    double final_a = omp_get_wtime();
    #pragma omp parallel shared(csr_nrows, csr_ncols, row_ptr, dloc)
    {
    int tid = omp_get_thread_num();
    VNT first_row = offsets[tid].first;
    VNT last_row = offsets[tid].second;
    Index* this_csc;

    for(VNT row = first_row; row < last_row; row++) {
        for (int j = csr_data->get_row_ptr()[row]; j < csr_data->get_row_ptr()[row + 1]; j++) {
            auto loc = row_ptr[csr_data->get_col_ids()[j]] + dloc[j];
            csc_data->get_col_ids()[loc] = row;
            csc_data->get_vals()[loc] = csr_data->get_vals()[j];
        }
    }
    }
    #pragma omp barrier
    double final_b = omp_get_wtime();

    if(max_threads > THREADS_PER_SOCKET)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void Matrix<T>::scantrans(void) {
    double mem_a = omp_get_wtime();
    memset(csc_data->get_row_ptr(),0, (csc_data->get_num_rows() + 1) * sizeof(Index));
    memset(csc_data->get_col_ids(),0, csc_data->get_nnz() * sizeof(Index));
    memset(csc_data->get_vals(),0, csc_data->get_nnz() * sizeof(T));

    Index* intra; Index* inter; VNT* csr_row_idx;
    int max_threads = omp_get_max_threads();

    ReducedWorkspace<T> csr_row_idx_ws(csc_data->get_nnz());
    ReducedWorkspace<T> intra_ws(csc_data->get_nnz());
    ReducedWorkspace<T> inter_ws((max_threads + 1) * csr_data->get_num_cols());

    auto ncols = csr_data->get_num_cols();
    auto nrows = csr_data->get_num_rows();

    std::cout << "Entering parallel region" << std::endl;

#pragma omp parallel num_threads(max_threads)
{
    int ithread = omp_get_thread_num();
    Index len; Index offset;

    if (ithread < csc_data->get_nnz() % max_threads) {
        len = csc_data->get_nnz() / max_threads + 1;
        offset = len * ithread;
    } else {
        len = csc_data->get_nnz() / max_threads;
        offset = (csc_data->get_nnz() % max_threads) * (len + 1) + (ithread -  csc_data->get_nnz() % max_threads) * len;
    }

    //std::cout << "Counted offsets " << len << " "<< offset << std::endl;

    Index start_index = 0;

    for (Index j = offset; j < offset + len; j++) {
        start_index = detect_row_idx(csr_data->get_row_ptr(), j, nrows, start_index);
        //std::cout << start_index << " on index " << j << std::endl;
        csr_row_idx_ws.get_element(j) = start_index;
        //std::cout << "got element " << std::endl;
    }

    for (Index i = 0; i < len; i++) {
        intra_ws.get_element(offset + i) =
                inter_ws.get_element((ithread + 1) * ncols + csr_data->get_col_ids()[offset + i])++;
    }
}

    std::cout << std::endl<< "Exited parallel region" << std::endl;


/*Vertical scan - maybe TODO*/
#pragma omp parallel for schedule(dynamic)
    for (VNT i = 0; i < ncols; i++) {
        for (VNT j = 1; j < max_threads + 1; j++) {
            inter_ws.get_element(i + ncols * j) += inter_ws.get_element(i + ncols * (j - 1));
        }
    }

    std::cout << "Exited vertical scan" << std::endl;
//    for (int i = 0; i < (max_threads + 1) * csr_data->get_num_cols(); i++) {
//        std::cout << inter_ws.get_element(i) << " ";
//    }
//    std::cout << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (VNT i = 0; i < ncols; i++) {
        csc_data->get_row_ptr()[i] = inter_ws.get_element(ncols * max_threads + i);
    }

    std::cout << "Exited copying " << std::endl;

    ParallelPrimitives::exclusive_scan(csc_data->get_row_ptr(), csc_data->get_row_ptr(), ncols);

    std::cout << "Exited exclusive scan " << std::endl;

//    for (int i = 0; i < csc_data->get_num_rows() + 1; i++) {
//        std::cout << csc_data->get_row_ptr()[i] << " ";
//    }
//    std::cout << std::endl;

#pragma omp parallel num_threads(max_threads)
    {
        int ithread = omp_get_thread_num();
        Index len; Index offset;

        if (ithread < csc_data->get_nnz() % max_threads) {
            len = csc_data->get_nnz() / max_threads + 1;
            offset = len * ithread;
        } else {
            len = csc_data->get_nnz() / max_threads;
            offset = (csc_data->get_nnz() % max_threads) * (len + 1) + (ithread -  csc_data->get_nnz() % max_threads) * len;
        }

        for (Index i = 0; i < len; i++) {
            auto loc = csc_data->get_row_ptr()[csr_data->get_col_ids()[offset + i]] +
                    inter_ws.get_element(ithread * ncols + csr_data->get_col_ids()[offset + i]) +
                    intra_ws.get_element(offset + i);
            csc_data->get_col_ids()[loc] = csr_row_idx_ws.get_element(offset + i);
            csc_data->get_vals()[loc] = csr_data->get_vals()[offset + i];
        }
    }
#ifdef __DEBUG_BANDWIDTHS__
    std::cout << "Inner time for mem " << mem_b - mem_a << " seconds" << std::endl;
    std::cout << "Inner time for fetch " << fetch_b - fetch_a << " seconds" << std::endl;
    std::cout << "Inner time for scan " << scan_b - scan_a << " seconds" << std::endl;
    std::cout << "Inner time for final " << final_b - final_a << " seconds" << std::endl;
#endif
}
#endif //GB_KUN_TRANSPOSE_HPP
