/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::construct_unsorted_csr(const VNT *_row_ids,
                                          const VNT *_col_ids,
                                          T *_vals,
                                          VNT _size,
                                          ENT _nz)
{
    vector<vector<VNT>> tmp_col_ids(_size);
    vector<vector<T>> tmp_vals(_size);

    for(ENT i = 0; i < _nz; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];
        tmp_col_ids[row].push_back(col);
        tmp_vals[row].push_back(val);
    }

    resize(_size, _nz);

    ENT cur_pos = 0;
    for(VNT i = 0; i < size; i++)
    {
        row_ptr[i] = cur_pos;
        row_ptr[i + 1] = cur_pos + tmp_col_ids[i].size();
        for(ENT j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            col_ids[j] = tmp_col_ids[i][j - row_ptr[i]];
            vals[j] = tmp_vals[i][j - row_ptr[i]];
        }
        cur_pos += tmp_col_ids[i].size();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::numa_aware_alloc()
{
    for(VNT i = 0; i < size + 1; i++)
    {
        row_ptr[i] = 0;
    }
    for(ENT i = 0; i < nz; i++)
    {
        vals[i] = 0;
        col_ids[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket)
{
    resize(_size, _nz);

    #pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        if(_socket == 0)
            if(tid == 0)
            {
                numa_aware_alloc();
            }
        if(_socket == 1)
            if(tid == (total_threads - 1))
            {
                numa_aware_alloc();
            }
    }

    construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
