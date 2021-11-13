/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::construct_unsorted_csr(const VNT *_row_ids,
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

bool cmp(pair<VNT, ENT>& a,
         pair<VNT, ENT>& b)
{
    return a.second > b.second;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::prepare_hub_data(map<int, int> &_freqs)
{
    vector<pair<VNT, ENT> > sorted_accesses;

    for (auto& it : _freqs) {
        sorted_accesses.push_back(it);
    }

    sort(sorted_accesses.begin(), sorted_accesses.end(), cmp);

    MemoryAPI::allocate_array(&hub_conversion_array, HUB_VERTICES);

    VNT *hub_positions;
    MemoryAPI::allocate_array(&hub_positions, size);
    for(VNT i = 0; i < size; i++)
    {
        hub_positions[i] = -1;
    }

    ENT cur_accesses = 0;
    for (VNT i = 0; i < HUB_VERTICES; i++)
    {
        cur_accesses += sorted_accesses[i].second;
        hub_conversion_array[i] = sorted_accesses[i].first;
        hub_positions[sorted_accesses[i].first] = i;
    }

    for (VNT i = 0; i < HUB_VERTICES; i++)
    {
        cout << "hub: " << i << " " << hub_conversion_array[i] << endl;
    }

    for(ENT i = 0; i < nz; i++)
    {
        VNT vertex = col_ids[i];
        if(hub_positions[vertex] >= 0)
        {
            col_ids[i] = hub_positions[col_ids[i]] * (-1);
        }
    }

    MemoryAPI::free_array(hub_positions);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz)
{
    resize(_size, _nz);

    construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _nz);

    map<VNT, ENT> col_freqs;
    map<VNT, ENT> row_freqs;

    for(ENT i = 0; i < _nz; i++)
    {
        VNT row_id = _row_ids[i];
        VNT col_id = _col_ids[i];
        col_freqs[col_id]++;
        row_freqs[row_id]++;
    }

    prepare_hub_data(col_freqs);

    for(ENT i = 0; i < _nz; i++)
    {
        cout << col_ids[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
