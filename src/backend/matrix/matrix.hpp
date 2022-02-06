#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(): _format(CSR)
{
    csr_data = NULL;
    csc_data = NULL;
    data = NULL;
    transposed_data = NULL;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    data_socket_dub = NULL;
    #endif

    workspace = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(Index ncols, Index nrows) : _format(CSR)
{
    throw "Error: Matrix(Index ncols, Index nrows) not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::~Matrix()
{
    if(csr_data != NULL)
        delete csr_data;

    if(csc_data != NULL)
        delete csc_data;

    if(data != NULL)
        delete data;
    if(transposed_data != NULL)
        delete transposed_data;

    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    if(data_socket_dub != NULL)
        delete data_socket_dub;
    #endif

    delete workspace;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_portion(FILE *_fp, VNT *_src_ids, VNT *_dst_ids, ENT _ln_pos, ENT _nnz)
{
    ENT end_pos = _ln_pos + MTX_READ_PARTITION_SIZE;
    for(size_t ln = _ln_pos; ln < min(_nnz, end_pos); ln++)
    {
        long long int src_id = -2, dst_id = -2;
        fscanf(_fp, "%lld %lld", &src_id, &dst_id);
        _src_ids[ln - _ln_pos] = src_id;
        _dst_ids[ln - _ln_pos] = dst_id;
        if(src_id <= 0 || dst_id <= 0)
            cout << "Error in read_portion, <= 0 src/dst ids" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void process_portion(const VNT *_src_ids,
                     const VNT *_dst_ids,
                     vector<vector<pair<VNT, T>>> &_csr_matrix,
                     vector<vector<pair<VNT, T>>> &_csc_matrix,
                     ENT _ln_pos,
                     ENT _nnz)
{
    unsigned int seed = int(time(NULL));
    for(size_t i = 0; i < MTX_READ_PARTITION_SIZE; i++)
    {
        if((_ln_pos + i) < _nnz)
        {
            VNT row = _src_ids[i] - 1;
            VNT col = _dst_ids[i] - 1;
            T val = EDGE_VAL;

            _csr_matrix[row].push_back(make_pair(col, val));
            _csc_matrix[col].push_back(make_pair(row, val));
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::read_mtx_file_pipelined(const string &_mtx_file_name,
                                        vector<vector<pair<VNT, T>>> &_csr_matrix,
                                        vector<vector<pair<VNT, T>>> &_csc_matrix)
{
    double t1, t2;
    t1 = omp_get_wtime();
    FILE *fp = fopen(_mtx_file_name.c_str(), "r");
    char header_line[4096];

    while(true)
    {
        fgets(header_line, 4096, fp);
        if(header_line[0] != '%')
            break;
    }

    long long int tmp_rows = 0, tmp_cols = 0, tmp_nnz = 0;
    sscanf(header_line, "%lld %lld %lld", &tmp_rows, &tmp_cols, &tmp_nnz);
    string hd_str(header_line);
    cout << "started reading, header is : " << hd_str << endl;

    VNT *proc_src_ids, *proc_dst_ids;
    VNT *read_src_ids, *read_dst_ids;
    MemoryAPI::allocate_array(&proc_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&proc_dst_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_dst_ids, MTX_READ_PARTITION_SIZE);

    ENT ln_pos = 0;

    _csr_matrix.resize(tmp_rows);
    _csc_matrix.resize(tmp_cols);

    #pragma omp parallel num_threads(2) shared(ln_pos)
    {
        int tid = omp_get_thread_num();

        if(tid == 0)
        {
            read_portion(fp, proc_src_ids, proc_dst_ids, ln_pos, (ENT)tmp_nnz);
            ln_pos += MTX_READ_PARTITION_SIZE;
        }

        while(ln_pos < tmp_nnz)
        {
            #pragma omp barrier

            if(tid == 0)
            {
                read_portion(fp, read_src_ids, read_dst_ids, ln_pos, (ENT)tmp_nnz);
            }
            if(tid == 1)
            {
                process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix,
                                ln_pos - MTX_READ_PARTITION_SIZE, tmp_nnz);
            }

            #pragma omp barrier

            if(tid == 0)
            {
                ptr_swap(read_src_ids, proc_src_ids);
                ptr_swap(read_dst_ids, proc_dst_ids);
                ln_pos += MTX_READ_PARTITION_SIZE;
            }

            #pragma omp barrier
        }

        if(tid == 1)
        {
            process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix, ln_pos - MTX_READ_PARTITION_SIZE, tmp_nnz);
        }

        #pragma omp barrier
    }

    MemoryAPI::free_array(proc_src_ids);
    MemoryAPI::free_array(proc_dst_ids);
    MemoryAPI::free_array(read_src_ids);
    MemoryAPI::free_array(read_dst_ids);

    fclose(fp);
    t2 = omp_get_wtime();
    cout << "CSR generation + C-style file read time: " << t2 - t1 << " sec" << endl;
}

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

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCSR<T>;
        ((MatrixCSR<T>*)data_socket_dub)->deep_copy((MatrixCSR<T>*)csr_data, 1);
        #endif
        cout << "Using CSR matrix format as optimized representation" << endl;
    }
    else if (_format == CSR_SEG)
    {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;
        ((MatrixSegmentedCSR<T>*)data)->build(csr_data->get_num_rows(), csr_data->get_nnz(), csr_data->get_row_ptr(), csr_data->get_col_ids(),
                    csr_data->get_vals(), 0);
        ((MatrixSegmentedCSR<T>*)transposed_data)->build(csc_data->get_num_rows(), csc_data->get_nnz(), csc_data->get_row_ptr(), csc_data->get_col_ids(),
                    csc_data->get_vals(), 0);

        data_socket_dub = NULL;
        cout << "Using CSR_SEG matrix format as optimized representation" << endl;
    }
    else if (_format == COO)
    {
        data = new MatrixCOO<T>;
        transposed_data = new MatrixCOO<T>;
        ((MatrixCOO<T>*)data)->build(csr_data->get_num_rows(), csr_data->get_nnz(), csr_data->get_row_ptr(), csr_data->get_col_ids(),
                                     csr_data->get_vals(), 0);
        ((MatrixCOO<T>*)transposed_data)->build(csc_data->get_num_rows(), csc_data->get_nnz(), csc_data->get_row_ptr(), csc_data->get_col_ids(),
                                                csc_data->get_vals(), 0);

        data_socket_dub = NULL;
        cout << "Using COO matrix format as optimized representation" << endl;
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

        data_socket_dub = NULL;
        cout << "Using SORTED CSR matrix format as optimized representation" << endl;
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

        data_socket_dub = NULL;
        cout << "Using SELL-C matrix format as optimized representation" << endl;
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

        data_socket_dub = NULL;
        cout << "Using LAV matrix format as optimized representation" << endl;
    }
    else
    {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    t2 = omp_get_wtime();
    cout << "creating optimized representation time: " << t2 - t1 << " sec" << endl;

    workspace = new Workspace(get_nrows(), get_ncols());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(const VNT *_row_indices,
                      const VNT *_col_indices,
                      const T *_values,
                      const VNT _size, // todo remove
                      const ENT _nnz)
{
    // CSR data creation
    double t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
    csc_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
    double t2 = omp_get_wtime();
    cout << "csr creation time: " << t2 - t1 << " sec" << endl;

    // initializing additional data structures time
    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::init_from_mtx(const string &_mtx_file_name)
{
    // read mtx file and get tmp representations of csr and csc matrix
    vector<vector<pair<VNT, T>>> csr_tmp_matrix;
    vector<vector<pair<VNT, T>>> csc_tmp_matrix;
    read_mtx_file_pipelined(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix);

    double t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(csr_tmp_matrix, 0);
    csc_data->build(csc_tmp_matrix, 0);
    double t2 = omp_get_wtime();
    cout << "csr (from mtx) creation time: " << t2 - t1 << " sec" << endl;

    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
