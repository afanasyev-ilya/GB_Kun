#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_portion(FILE *_fp, VNT *_src_ids, VNT *_dst_ids, ENT _ln_pos, ENT _nnz)
{
    ENT end_pos = _ln_pos + MTX_READ_PARTITION_SIZE;
    const int buffer_size = 8192;
    char buffer[buffer_size];
    for (size_t ln = _ln_pos; ln < min(_nnz, end_pos); ln++)
    {
        long long int src_id = -2, dst_id = -2;
        if (fgets(buffer, buffer_size, _fp) == NULL)
            throw "Error: unexpected end of graph file! Aborting...";

        sscanf(buffer, "%lld %lld", &src_id, &dst_id);
        _src_ids[ln - _ln_pos] = src_id;
        _dst_ids[ln - _ln_pos] = dst_id;
        if (src_id <= 0 || dst_id <= 0)
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
    for (size_t i = 0; i < MTX_READ_PARTITION_SIZE; i++)
    {
        if ((_ln_pos + i) < _nnz)
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
    if (fp == 0)
    {
        cout << "Error: Can not open .mtx " << _mtx_file_name.c_str() << " file!" << endl;
        throw "Error: Can not open .mtx file";
    }

    char header_line[4096];

    while (true)
    {
        if (fgets(header_line, 4096, fp) == NULL)
            throw "Error: unexpected end of graph file! Aborting...";
        if (header_line[0] != '%')
            break;
    }

    long long int tmp_rows = 0, tmp_cols = 0, tmp_nnz = 0;
    sscanf(header_line, "%lld %lld %lld", &tmp_rows, &tmp_cols, &tmp_nnz);
    string hd_str(header_line);
    cout << "started reading, header is : " << hd_str << endl;

    VNT * proc_src_ids, *proc_dst_ids;
    VNT * read_src_ids, *read_dst_ids;
    MemoryAPI::allocate_array(&proc_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&proc_dst_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_dst_ids, MTX_READ_PARTITION_SIZE);

    ENT ln_pos = 0;

    _csr_matrix.resize(tmp_rows);
    _csc_matrix.resize(tmp_cols);

    int min_seq_steps = 8;

    if (tmp_nnz >= MTX_READ_PARTITION_SIZE * min_seq_steps)
    {
        #pragma omp parallel num_threads(2) shared(ln_pos)
        {
            int tid = omp_get_thread_num();

            if (tid == 0)
            {
                read_portion(fp, proc_src_ids, proc_dst_ids, ln_pos, (ENT)
                tmp_nnz);
                ln_pos += MTX_READ_PARTITION_SIZE;
            }

            #pragma omp barrier

            while (ln_pos < tmp_nnz)
            {
                #pragma omp barrier

                if (tid == 0)
                {
                    read_portion(fp, read_src_ids, read_dst_ids, ln_pos, (ENT)
                    tmp_nnz);
                }
                if (tid == 1)
                {
                    process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix,
                                    ln_pos - MTX_READ_PARTITION_SIZE, tmp_nnz);
                }

                #pragma omp barrier

                if (tid == 0)
                {
                    ptr_swap(read_src_ids, proc_src_ids);
                    ptr_swap(read_dst_ids, proc_dst_ids);
                    ln_pos += MTX_READ_PARTITION_SIZE;
                }

                #pragma omp barrier
            }

            if (tid == 1)
            {
                process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix, ln_pos - MTX_READ_PARTITION_SIZE,
                                tmp_nnz);
            }

            #pragma omp barrier
        }
    } else
    {
        ln_pos = 0;
        while (ln_pos < tmp_nnz)
        {
            read_portion(fp, proc_src_ids, proc_dst_ids, ln_pos, (ENT)
            tmp_nnz);
            process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix, ln_pos, tmp_nnz);
            ln_pos += MTX_READ_PARTITION_SIZE;
        }
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

void binary_read_portion(FILE *_fp, VNT *_src_ids, VNT *_dst_ids, ENT _ln_pos, ENT _nnz)
{
    ENT end_pos = _ln_pos + MTX_READ_PARTITION_SIZE;

    VNT buf_size = MTX_READ_PARTITION_SIZE * 2 * sizeof(VNT);
    VNT *buf = (VNT * )
    malloc(buf_size);
    if (fread(buf, sizeof(VNT), 2 * MTX_READ_PARTITION_SIZE, _fp) == 0)
        throw "Error! Unexpected end of binary file";
    for (size_t ln = _ln_pos, i = 0; ln < min(_nnz, end_pos); ln++, i += 2)
    {
        VNT src_id = -2, dst_id = -2;
        src_id = buf[i];
        dst_id = buf[i + 1];
        _src_ids[ln - _ln_pos] = src_id;
        _dst_ids[ln - _ln_pos] = dst_id;
        if (src_id <= 0 || dst_id <= 0)
            cout << "Error in read_portion, <= 0 src/dst ids" << endl;
    }
    free(buf);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::binary_read_mtx_file_pipelined(const string &_mtx_file_name,
                                               vector<vector<pair<VNT, T>>> &_csr_matrix,
                                               vector<vector<pair<VNT, T>>> &_csc_matrix)
{
    double t1, t2;
    t1 = omp_get_wtime();
    FILE *fp = fopen(_mtx_file_name.c_str(), "rb");
    if (fp == 0)
    {
        cout << "Error: Can not open .mtx " << _mtx_file_name.c_str() << " file!" << endl;
        throw "Error: Can not open .mtx file";
    }

    long long int tmp_rows = 0, tmp_cols = 0, tmp_nnz = 0;
    if (fread(&tmp_rows, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";
    if (fread(&tmp_cols, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";
    if (fread(&tmp_nnz, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";

    VNT * proc_src_ids, *proc_dst_ids;
    VNT * read_src_ids, *read_dst_ids;
    MemoryAPI::allocate_array(&proc_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&proc_dst_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_src_ids, MTX_READ_PARTITION_SIZE);
    MemoryAPI::allocate_array(&read_dst_ids, MTX_READ_PARTITION_SIZE);

    ENT ln_pos = 0;

    _csr_matrix.resize(tmp_rows);
    _csc_matrix.resize(tmp_cols);

    int min_seq_steps = 8;

    if (tmp_nnz >= MTX_READ_PARTITION_SIZE * min_seq_steps)
    {
        #pragma omp parallel num_threads(2) shared(ln_pos)
        {
            int tid = omp_get_thread_num();

            if (tid == 0)
            {
                binary_read_portion(fp, proc_src_ids, proc_dst_ids, ln_pos, (ENT)
                tmp_nnz);
                ln_pos += MTX_READ_PARTITION_SIZE;
            }

            #pragma omp barrier

            while (ln_pos < tmp_nnz)
            {
                #pragma omp barrier

                if (tid == 0)
                {
                    binary_read_portion(fp, read_src_ids, read_dst_ids, ln_pos, (ENT)
                    tmp_nnz);
                }
                if (tid == 1)
                {
                    process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix,
                                    ln_pos - MTX_READ_PARTITION_SIZE, tmp_nnz);
                }

                #pragma omp barrier

                if (tid == 0)
                {
                    ptr_swap(read_src_ids, proc_src_ids);
                    ptr_swap(read_dst_ids, proc_dst_ids);
                    ln_pos += MTX_READ_PARTITION_SIZE;
                }

                #pragma omp barrier
            }

            if (tid == 1)
            {
                process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix, ln_pos - MTX_READ_PARTITION_SIZE,
                                tmp_nnz);
            }

            #pragma omp barrier
        }
    } else
    {
        ln_pos = 0;
        while (ln_pos < tmp_nnz)
        {
            binary_read_portion(fp, proc_src_ids, proc_dst_ids, ln_pos, (ENT)
            tmp_nnz);
            process_portion(proc_src_ids, proc_dst_ids, _csr_matrix, _csc_matrix, ln_pos, tmp_nnz);
            ln_pos += MTX_READ_PARTITION_SIZE;
        }
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
void Matrix<T>::read_mtx_file_sequential(const string &_mtx_file_name,
                                         vector<vector<pair<VNT, T>>> &_csr_matrix,
                                         vector<vector<pair<VNT, T>>> &_csc_matrix)
{
    FILE *fp = fopen(_mtx_file_name.c_str(), "r");
    if (fp == 0)
    {
        cout << "Error: Can not open .mtx " << _mtx_file_name.c_str() << " file!" << endl;
        throw "Error: Can not open .mtx file";
    }

    char header_line[4096];

    while (true)
    {
        if (fgets(header_line, 4096, fp) == NULL)
            throw "Error: unexpected end of graph file! Aborting...";
        if (header_line[0] != '%')
            break;
    }

    long long int tmp_rows = 0, tmp_cols = 0, tmp_nnz = 0;
    sscanf(header_line, "%lld %lld %lld", &tmp_rows, &tmp_cols, &tmp_nnz);
    string hd_str(header_line);

    const int buffer_size = 8192;
    char buffer[buffer_size];

    _csr_matrix.resize(tmp_rows);
    _csc_matrix.resize(tmp_cols);

    for(long long int i = 0; i < tmp_nnz; i++)
    {
        long long int src_id = -2, dst_id = -2;
        if (fgets(buffer, buffer_size, fp) == NULL)
            throw "Error: unexpected end of graph file! Aborting...";

        sscanf(buffer, "%lld %lld", &src_id, &dst_id);
        if (src_id <= 0 || dst_id <= 0)
            cout << "Error in read_portion, <= 0 src/dst ids" << endl;

        VNT row = src_id - 1;
        VNT col = dst_id - 1;
        T val = EDGE_VAL;

        _csr_matrix[row].push_back(make_pair(col, val));
        _csc_matrix[col].push_back(make_pair(row, val));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::binary_read_mtx_file(const string &_mtx_file_name,
                                     vector<vector<pair<VNT, T>>> &_csr_matrix,
                                     vector<vector<pair<VNT, T>>> &_csc_matrix)
{
    FILE *fp = fopen(_mtx_file_name.c_str(), "rb");
    if (fp == 0)
    {
        cout << "Error: Can not open .mtx " << _mtx_file_name.c_str() << " file!" << endl;
        throw "Error: Can not open .mtx file";
    }

    long long int nrows = 0, ncols = 0, nnz = 0;
    if (fread(&nrows, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";
    if (fread(&ncols, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";
    if (fread(&nnz, sizeof(long long), 1, fp) == 0)
        throw "Error! Unexpected end of binary file";

    std::vector<VNT> all_data_vec(nnz*2, 0);

    {
        Timer tm("new binary read");

        fread(&all_data_vec[0], sizeof(VNT), nnz*2, fp);
    }

    _csr_matrix.resize(nrows);
    _csc_matrix.resize(ncols);
    {
        Timer tm("graph creation time");
        //#pragma omp parallel for num_threads(creation_threads)
        for(ENT i = 0; i < nnz; i++)
        {
            VNT src_id = all_data_vec[2*i];
            VNT dst_id = all_data_vec[2*i + 1];
            T val = EDGE_VAL;
            _csr_matrix[src_id].push_back(std::make_pair(dst_id, val));
            _csc_matrix[dst_id].push_back(std::make_pair(src_id, val));
        }
    }

    fclose(fp);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::init_from_mtx(const string &_mtx_file_name)
{
    // read mtx file and get tmp representations of csr and csc matrix
    vector<vector<pair<VNT, T>>> csr_tmp_matrix;
    vector<vector<pair<VNT, T>>> csc_tmp_matrix;
    if(ends_with(_mtx_file_name, "mtx"))
    {
        #ifdef __DEBUG_FILE_IO__
        SAVE_TIME_SEC((read_mtx_file_pipelined(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix)), "mtx_read");
        #else
        read_mtx_file_pipelined(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix);
        #endif
    }
    else if(ends_with(_mtx_file_name, "mtxbin"))
    {
        //binary_read_mtx_file(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix);

        {
            Timer tm("old binary read");
            #ifdef __DEBUG_FILE_IO__
            SAVE_TIME_SEC((binary_read_mtx_file_pipelined(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix)), "binary_read");
            #else
            binary_read_mtx_file_pipelined(_mtx_file_name, csr_tmp_matrix, csc_tmp_matrix);
            #endif
        }
    }
    else
    {
        cout << "Unsupported matrix file format. can be either .mtx or .mtx.bin";
        throw "Aborting...";
    }

    VNT tmp_nrows = csr_tmp_matrix.size(), tmp_ncols = csc_tmp_matrix.size();

    double t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(csr_tmp_matrix, tmp_nrows, tmp_ncols);
    csc_data->build(csc_tmp_matrix, tmp_ncols, tmp_nrows);
    double t2 = omp_get_wtime();

    #ifdef __DEBUG_FILE_IO__
    save_time_in_sec("build_matrix_from_file", t2 - t1);
    #endif

    #ifdef __DEBUG_INFO__
    cout << "csr (from mtx) creation time: " << t2 - t1 << " sec" << endl;
    #endif

    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
