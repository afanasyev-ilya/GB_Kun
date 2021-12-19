/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::construct_unsorted_csr(const VNT *_row_ids,
                                          const VNT *_col_ids,
                                          const T *_vals,
                                          VNT _size,
                                          ENT _nnz)
{
    vector<vector<VNT>> tmp_col_ids(_size);
    vector<vector<T>> tmp_vals(_size);

    for(ENT i = 0; i < _nnz; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];
        tmp_col_ids[row].push_back(col);
        tmp_vals[row].push_back(val);
    }

    resize(_size, _nnz);

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
void MatrixCSR<T>::numa_aware_alloc()
{
    for(VNT i = 0; i < size + 1; i++)
    {
        row_ptr[i] = 0;
    }
    for(ENT i = 0; i < nnz; i++)
    {
        vals[i] = 0;
        col_ids[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::prepare_vg_lists()
{
    /*vertex_groups[0].set_thresholds(0, 8);
    vertex_groups[1].set_thresholds(8, 16);
    vertex_groups[2].set_thresholds(16, 64);
    vertex_groups[3].set_thresholds(64, 128);
    vertex_groups[4].set_thresholds(128, 256);
    vertex_groups[5].set_thresholds(256, INT_MAX);*/

    ENT step = 16;
    ENT first = 4;
    vertex_groups[0].set_thresholds(0, first);
    for(int i = 1; i < (vg_num - 1); i++)
    {
        vertex_groups[i].set_thresholds(first, first*step);
        first *= step;
    }
    vertex_groups[vg_num - 1].set_thresholds(first, INT_MAX);

    /*vertex_groups[0].set_thresholds(0, 4);
    vertex_groups[1].set_thresholds(4, 8);
    vertex_groups[2].set_thresholds(8, 16);
    vertex_groups[3].set_thresholds(16, 32);
    vertex_groups[4].set_thresholds(32, 64);
    vertex_groups[5].set_thresholds(64, 128);
    vertex_groups[6].set_thresholds(128, 256);
    vertex_groups[7].set_thresholds(256, 512);
    vertex_groups[8].set_thresholds(512, INT_MAX);*/

    for(VNT row = 0; row < size; row++)
    {
        ENT connections_count = row_ptr[row + 1] - row_ptr[row];
        for(int vg = 0; vg < vg_num; vg++)
            if(vertex_groups[vg].in_range(connections_count))
                vertex_groups[vg].push_back(row);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    resize(_size, _nnz);

    /*#pragma omp parallel
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
    }*/

    construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _nnz);

    prepare_vg_lists();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
