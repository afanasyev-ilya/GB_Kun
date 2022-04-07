/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::construct_unsorted_csr(vector<vector<VNT>> &_tmp_col_ids,
                                          vector<vector<T>> &_tmp_vals,
                                          LAVSegment<T> *_cur_segment,
                                          ENT _total_nnz)
{
    ENT local_size = _tmp_col_ids.size();
    VNT zero_rows = 0;
    ENT local_nnz = 0;
    #pragma omp parallel for reduction(+: local_nnz)
    for(VNT i = 0; i < local_size; i++)
    {
        local_nnz += _tmp_col_ids[i].size();
        if(_tmp_col_ids[i].size() == 0)
            zero_rows++;
    }
    cout << "local nnz: " << local_nnz << ", " << 100.0*((double)local_nnz/_total_nnz) << " % (of total)" << endl;
    cout << "average degree: " << ((double)local_nnz / local_size) << endl;
    cout << "zero rows: " << 100.0*((double)zero_rows / local_size) << " %" << endl;

    _cur_segment->vertex_list.set_thresholds(0, INT_MAX);
    for(VNT row = 0; row < local_size; row++)
    {
        if(_tmp_col_ids[row].size() > 0)
            _cur_segment->vertex_list.push_back(row);
    }

    _cur_segment->vertex_list.finalize_creation(0 /* TODO */);

    MemoryAPI::allocate_array(&(_cur_segment->row_ptr), local_size + 1);
    MemoryAPI::allocate_array(&(_cur_segment->col_ids), local_nnz);
    MemoryAPI::allocate_array(&(_cur_segment->vals), local_nnz);

    _cur_segment->min_col_id = INT_MAX;
    _cur_segment->max_col_id = 0;

    ENT cur_pos = 0;
    for(VNT i = 0; i < local_size; i++)
    {
        (_cur_segment->row_ptr)[i] = cur_pos;
        (_cur_segment->row_ptr)[i + 1] = cur_pos + _tmp_col_ids[i].size();
        for(ENT j = (_cur_segment->row_ptr)[i]; j < (_cur_segment->row_ptr)[i + 1]; j++)
        {
            (_cur_segment->col_ids)[j] = _tmp_col_ids[i][j - (_cur_segment->row_ptr)[i]];
            (_cur_segment->vals)[j] = _tmp_vals[i][j - (_cur_segment->row_ptr)[i]];

            VNT col_id = (_cur_segment->col_ids)[j];

            if(col_id < _cur_segment->min_col_id)
                _cur_segment->min_col_id = col_id;
            if(col_id > _cur_segment->max_col_id)
                _cur_segment->max_col_id = col_id;
        }
        cur_pos += _tmp_col_ids[i].size();
    }

    _cur_segment->nnz = local_nnz;
    _cur_segment->size = local_size;

    cout << "segment ids in range of: " << (_cur_segment->max_col_id - _cur_segment->min_col_id)*sizeof(T) / 1e3 << " KB" << endl;
    cout << "starting: " << _cur_segment->min_col_id << " ending: " << _cur_segment->max_col_id << endl << endl;

    ENT step = 16;
    ENT first = 4;
    _cur_segment->vertex_groups[0].set_thresholds(0, first);
    for(int i = 1; i < (_cur_segment->vg_num - 1); i++)
    {
        _cur_segment->vertex_groups[i].set_thresholds(first, first*step);
        first *= step;
    }
    _cur_segment->vertex_groups[_cur_segment->vg_num - 1].set_thresholds(first, INT_MAX);

    for(VNT row = 0; row < local_size; row++)
    {
        ENT connections_count = _tmp_col_ids[row].size();
        if(connections_count > 0)
        {
            for(int vg = 0; vg < _cur_segment->vg_num; vg++)
                if(_cur_segment->vertex_groups[vg].in_range(connections_count))
                    _cur_segment->vertex_groups[vg].push_back(row);
        }
    }

    for(int vg = 0; vg < _cur_segment->vg_num; vg++)
    {
        _cur_segment->vertex_groups[vg].finalize_creation(0 /* TODO */);
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
void MatrixLAV<T>::build(VNT *_row_degrees,
                         VNT *_col_degrees,
                         VNT _nrows,
                         VNT _ncols,
                         ENT _nnz,
                         const ENT *_row_ptr,
                         const VNT *_col_ids,
                         const T *_vals,
                         int _target_socket)
{
    this->nrows = _nrows;
    this->ncols = _ncols;
    this->nnz = _nnz;
    cout << this->nrows << " " << this->ncols << " " << this->nnz << endl;

    MemoryAPI::allocate_array(&col_new_to_old, ncols);
    MemoryAPI::allocate_array(&col_old_to_new, ncols);

    #pragma omp parallel for
    for(VNT i = 0; i < _ncols; i++)
    {
        col_new_to_old[i] = i;
    }

    std::sort(col_new_to_old, col_new_to_old + ncols,
              [_col_degrees](VNT index1, VNT index2)
              {
                  return _col_degrees[index1] > _col_degrees[index2];
              });

    /*cout << "original col degrees: ";
    for(int i = 0; i < ncols; i++)
        cout << _col_degrees[i] << " ";
    cout << endl;

    cout << "sorted col degrees: ";
    for(int i = 0; i < ncols; i++)
        cout << _col_degrees[col_new_to_old[i]] << " "; // i-th current was at col_new_to_old[i]
    cout << endl;*/

    ENT nnz_cnt = 0;
    VNT dense_threshold = 0;
    for(VNT col = 0; col < ncols; col++)
    {
        nnz_cnt += _col_degrees[col_new_to_old[col]];
        if(nnz_cnt >= 0.8*_nnz)
        {
            dense_threshold = col;
            break;
        }
    }

    cout << "dense threshold: " << dense_threshold << " / " << _ncols << endl;
    VNT seg_size = 512*1024/sizeof(T);
    dense_segments_num = (dense_threshold - 1)/seg_size + 1;
    cout << "dense segments: " << dense_segments_num << endl;

    for(VNT i = 0; i < ncols; i++)
    {
        col_old_to_new[col_new_to_old[i]] = i;
    }

    vector<vector<vector<VNT>>> vec_dense_col_ids(dense_segments_num);
    vector<vector<vector<T>>> vec_dense_vals(dense_segments_num);

    for(VNT seg = 0; seg < dense_segments_num; seg++)
    {
        vec_dense_col_ids[seg].resize(nrows);
        vec_dense_vals[seg].resize(nrows);
    }

    vector<vector<VNT>> vec_sparse_col_ids(nrows);
    vector<vector<T>> vec_sparse_vals(nrows);

    for(VNT row = 0; row < nrows; row++)
    {
        for(ENT j = _row_ptr[row]; j < _row_ptr[row + 1]; j++)
        {
            VNT col = _col_ids[j];
            T val = _vals[j];

            VNT new_col = col_old_to_new[col];

            if(new_col < dense_threshold)
            {
                VNT seg_id = new_col / seg_size;

                vec_dense_col_ids[seg_id][row].push_back(new_col);
                vec_dense_vals[seg_id][row].push_back(val);
            }
            else
            {
                vec_sparse_col_ids[row].push_back(new_col);
                vec_sparse_vals[row].push_back(val);
            }
        }
    }

    dense_segments = new LAVSegment<T>[dense_segments_num];

    for(VNT seg = 0; seg < dense_segments_num; seg++)
    {
        vec_dense_col_ids[seg].resize(nrows);
        vec_dense_vals[seg].resize(nrows);

        construct_unsorted_csr(vec_dense_col_ids[seg], vec_dense_vals[seg], &(dense_segments[seg]), _nnz);
    }

    construct_unsorted_csr(vec_sparse_col_ids, vec_sparse_vals, &sparse_segment, _nnz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
