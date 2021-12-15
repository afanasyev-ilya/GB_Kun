/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::construct_unsorted_csr(vector<vector<VNT>> &_tmp_col_ids,
                                          vector<vector<T>> &_tmp_vals,
                                          ENT **local_row_ptr,
                                          VNT **local_col_ids,
                                          T **local_vals)
{
    ENT local_size = _tmp_col_ids.size();
    ENT local_nnz = 0;
    for(VNT i = 0; i < local_size; i++)
        local_nnz += _tmp_col_ids[i].size();
    cout << "local nnz: " << local_nnz << endl;

    MemoryAPI::allocate_array(local_row_ptr, local_size + 1);
    MemoryAPI::allocate_array(local_col_ids, local_nnz);
    MemoryAPI::allocate_array(local_vals, local_nnz);

    ENT cur_pos = 0;
    for(VNT i = 0; i < local_size; i++)
    {
        (*local_row_ptr)[i] = cur_pos;
        (*local_row_ptr)[i + 1] = cur_pos + _tmp_col_ids[i].size();
        for(ENT j = (*local_row_ptr)[i]; j < (*local_row_ptr)[i + 1]; j++)
        {
            (*local_col_ids)[j] = _tmp_col_ids[i][j - (*local_row_ptr)[i]];
            (*local_vals)[j] = _tmp_vals[i][j - (*local_row_ptr)[i]];
        }
        cur_pos += _tmp_col_ids[i].size();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool cmp(pair<VNT, ENT>& a,
         pair<VNT, ENT>& b)
{
    return a.second > b.second;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    size = _size;
    nnz = _nnz;

    cout << "starting build" << endl;

    map<VNT, ENT> col_freqs;
    map<VNT, ENT> row_freqs;

    for(ENT i = 0; i < _nnz; i++)
    {
        VNT row_id = _row_ids[i];
        VNT col_id = _col_ids[i];
        col_freqs[col_id]++;
        row_freqs[row_id]++;
    }

    cout << "freqs calculated" << endl;

    VNT *new_to_old, *old_to_new;
    VNT *cols_frequencies;
    MemoryAPI::allocate_array(&new_to_old, _size);
    MemoryAPI::allocate_array(&old_to_new, _size);
    MemoryAPI::allocate_array(&cols_frequencies, _size);
    for(VNT i = 0; i < _size; i++) {
        new_to_old[i] = i;
        cols_frequencies[i] = col_freqs[i];
    }

    std::sort(new_to_old, new_to_old + _size,
              [cols_frequencies](int index1, int index2)
              {
                  return cols_frequencies[index1] > cols_frequencies[index2];
              });

    ENT nnz_cnt = 0;
    VNT dense_threshold = 0;
    for(VNT col = 0; col < _size; col++)
    {
        nnz_cnt += cols_frequencies[new_to_old[col]];
        if(nnz_cnt >= 0.8*_nnz)
        {
            dense_threshold = col;
            break;
        }
    }

    cout << "dense threshold: " << dense_threshold << " / " << _size << endl;
    VNT seg_size = 512*1024/sizeof(T);
    dense_segments = (dense_threshold - 1)/seg_size + 1;
    cout << "dense segments: " << dense_segments << endl;

    for(VNT i = 0; i < size; i++)
    {
        old_to_new[new_to_old[i]] = i;
    }

    vector<vector<vector<VNT>>> vec_dense_col_ids(dense_segments);
    vector<vector<vector<T>>> vec_dense_vals(dense_segments);

    for(VNT seg = 0; seg < dense_segments; seg++)
    {
        vec_dense_col_ids[seg].resize(_size);
        vec_dense_vals[seg].resize(_size);
    }

    vector<vector<VNT>> vec_sparse_col_ids(_size);
    vector<vector<T>> vec_sparse_vals(_size);

    for(ENT i = 0; i < _nnz; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];

        VNT new_col = old_to_new[col];
        VNT seg_id = new_col / seg_size;

        if(new_col < dense_threshold)
        {
            vec_dense_col_ids[seg_id][row].push_back(new_col);
            vec_dense_vals[seg_id][row].push_back(val);
        }
        else
        {
            vec_sparse_col_ids[row].push_back(new_col);
            vec_sparse_vals[row].push_back(val);
        }
    }

    cout << "vectors prepared" << endl;

    dense_row_ptr = new ENT*[dense_segments];
    dense_col_ids = new VNT*[dense_segments];
    dense_vals = new T*[dense_segments];

    for(VNT seg = 0; seg < dense_segments; seg++)
    {
        vec_dense_col_ids[seg].resize(_size);
        vec_dense_vals[seg].resize(_size);

        construct_unsorted_csr(vec_dense_col_ids[seg], vec_dense_vals[seg], &(dense_row_ptr[seg]), &(dense_col_ids[seg]),
                               &(dense_vals[seg]));
    }

    construct_unsorted_csr(vec_sparse_col_ids, vec_sparse_vals, &sparse_row_ptr, &sparse_col_ids,
                           &sparse_vals);

    cout << "all csrs constructed" << endl;

    //construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _nnz);
    //prepare_hub_data(col_freqs);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
