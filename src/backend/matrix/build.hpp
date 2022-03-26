#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::init_optimized_structures()
{
    double t1, t2;

    t1 = omp_get_wtime();
    // optimized representation creation
    if (_format == CSR)
    {
        data = NULL;
        transposed_data = NULL;
        #ifdef __DEBUG_INFO__
        cout << "Using CSR matrix format as optimized representation" << endl;
        #endif
    }
    else if (_format == CSR_SEG)
    {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;
        ((MatrixSegmentedCSR<T>*)data)->build(csr_data->get_num_rows(), csr_data->get_nnz(), csr_data->get_row_ptr(), csr_data->get_col_ids(),
                                              csr_data->get_vals(), 0);
        ((MatrixSegmentedCSR<T>*)transposed_data)->build(csc_data->get_num_rows(), csc_data->get_nnz(), csc_data->get_row_ptr(), csc_data->get_col_ids(),
                                                         csc_data->get_vals(), 0);
        #ifdef __DEBUG_INFO__
        cout << "Using CSR_SEG matrix format as optimized representation" << endl;
        #endif
    }
    else if (_format == COO)
    {
        data = new MatrixCOO<T>;
        transposed_data = new MatrixCOO<T>;
        ((MatrixCOO<T>*)data)->build(csr_data->get_num_rows(), csr_data->get_nnz(), csr_data->get_row_ptr(), csr_data->get_col_ids(),
                                     csr_data->get_vals(), 0);
        ((MatrixCOO<T>*)transposed_data)->build(csc_data->get_num_rows(), csc_data->get_nnz(), csc_data->get_row_ptr(), csc_data->get_col_ids(),
                                                csc_data->get_vals(), 0);
        #ifdef __DEBUG_INFO__
        cout << "Using COO matrix format as optimized representation" << endl;
        #endif
    }
    else if (_format == SORTED_CSR)
    {
        data = new MatrixSortCSR<T>;
        transposed_data = new MatrixSortCSR<T>;
        ((MatrixSortCSR<T>*)data)->build(get_rowdegrees(), get_coldegrees(),
                                         csr_data->get_num_rows(),
                                         csr_data->get_num_cols(),
                                         csr_data->get_nnz(),
                                         csr_data->get_row_ptr(),
                                         csr_data->get_col_ids(),
                                         csr_data->get_vals(), 0);
        ((MatrixSortCSR<T>*)transposed_data)->build(get_coldegrees(), get_rowdegrees(),
                                                    csc_data->get_num_rows(),
                                                    csc_data->get_num_cols(),
                                                    csc_data->get_nnz(),
                                                    csc_data->get_row_ptr(),
                                                    csc_data->get_col_ids(),
                                                    csc_data->get_vals(), 0);
        #ifdef __DEBUG_INFO__
        cout << "Using SORTED CSR matrix format as optimized representation" << endl;
        #endif
    }
    else if (_format == SELL_C)
    {
        data = new MatrixSellC<T>;
        transposed_data = new MatrixSellC<T>;
        ((MatrixSellC<T>*)data)->build(csr_data->get_num_rows(),
                                       csr_data->get_num_cols(),
                                       csr_data->get_nnz(),
                                       csr_data->get_row_ptr(),
                                       csr_data->get_col_ids(),
                                       csr_data->get_vals(), 0);
        ((MatrixSellC<T>*)transposed_data)->build(csc_data->get_num_rows(), // since CSC is used no swap
                                                  csc_data->get_num_cols(), // compared to prev build
                                                  csc_data->get_nnz(),
                                                  csc_data->get_row_ptr(),
                                                  csc_data->get_col_ids(),
                                                  csc_data->get_vals(), 0);
        #ifdef __DEBUG_INFO__
        cout << "Using SELL-C matrix format as optimized representation" << endl;
        #endif
    }
    else if (_format == LAV)
    {
        data = new MatrixLAV<T>;
        transposed_data = new MatrixLAV<T>;
        ((MatrixLAV<T>*)data)->build(get_rowdegrees(), get_coldegrees(),
                                     csr_data->get_num_rows(),
                                     csr_data->get_num_cols(),
                                     csr_data->get_nnz(),
                                     csr_data->get_row_ptr(),
                                     csr_data->get_col_ids(),
                                     csr_data->get_vals(), 0);
        ((MatrixLAV<T>*)transposed_data)->build(get_coldegrees(), get_rowdegrees(),
                                                csc_data->get_num_rows(), // since CSC is used no swap
                                                csc_data->get_num_cols(), // compared to prev build
                                                csc_data->get_nnz(),
                                                csc_data->get_row_ptr(),
                                                csc_data->get_col_ids(),
                                                csc_data->get_vals(), 0);
        #ifdef __DEBUG_INFO__
        cout << "Using LAV matrix format as optimized representation" << endl;
        #endif
    }
    else
    {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    t2 = omp_get_wtime();
    #ifdef __DEBUG_INFO__
    cout << "creating optimized representation time: " << t2 - t1 << " sec" << endl;
    #endif

    workspace = new Workspace(get_nrows(), get_ncols());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(const VNT *_row_indices,
                      const VNT *_col_indices,
                      const T *_values,
                      ENT _nnz)
{
    VNT max_rows = 0, max_cols = 0;
    #pragma omp parallel for reduction(max: max_rows, max_cols)
    for(ENT i = 0; i < _nnz; i++)
    {
        if(max_rows < _row_indices[i])
        {
            max_rows = _row_indices[i];
        }

        if(max_cols < _col_indices[i])
        {
            max_cols = _col_indices[i];
        }
    }

    max_rows += 1;
    max_cols += 1;
    //std::cout << "MAX COLS " << max_cols << " and MAX_ROWS " << max_rows << std::endl;
    if(max_rows != max_cols)
    {
        cout << "Non-square matrix is not supported yet" << endl;
        VNT max_dim = max(max_rows, max_cols);
        max_rows = max_dim;
        max_cols = max_dim;
    }

    double t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(_row_indices, _col_indices, _values, max_rows, max_cols, _nnz);
    csc_data->build(_col_indices, _row_indices, _values, max_cols, max_rows, _nnz);
    double t2 = omp_get_wtime();
    #ifdef __DEBUG_INFO__
    cout << "csr creation time: " << t2 - t1 << " sec" << endl;
    #endif

    // initializing additional data structures time
    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build_from_csr_arrays(const ENT* _row_ptrs,
                                      const VNT *_col_ids,
                                      const T *_values,
                                      Index _nrows,
                                      Index _nnz)
{
    VNT max_rows = _nrows, max_cols = 0;
    #pragma omp parallel for reduction(max: max_cols)
    for(ENT i = 0; i < _nnz; i++)
    {
        if(max_cols < _col_ids[i])
        {
            max_cols = _col_ids[i];
        }
    }
    max_cols += 1; // to correct process last col correctly

    if(max_rows != max_cols)
    {
        cout << "Non-square matrix is not supported yet" << endl;
        VNT max_dim = max(max_rows, max_cols);
        max_rows = max_dim;
        max_cols = max_dim;
    }

    vector<ENT> extended_ptrs(max_rows + 1, _nnz); // this one is needed since number of rows could be increased before
    MemoryAPI::copy(&extended_ptrs[0], _row_ptrs, _nrows + 1);

    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build_from_csr_arrays(&extended_ptrs[0], _col_ids, _values, max_rows, max_cols, _nnz);
    csc_data->resize(max_cols, max_rows, _nnz);
    double t1 = omp_get_wtime();
    csr_to_csc();
    double t2 = omp_get_wtime();
    save_teps("transpose", t2 - t1, _nnz);

    // initializing additional data structures time
    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

