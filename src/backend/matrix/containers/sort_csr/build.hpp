/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSortCSR<T>::construct_csr(const VNT *_row_ids,
                                     const VNT *_col_ids,
                                     const T *_vals,
                                     VNT _size,
                                     ENT _nnz,
                                     int _target_socket)
{
    size = _size;
    nnz = _nnz;
    vector<vector<VNT>> tmp_col_ids(_size);
    vector<vector<T>> tmp_vals(_size);

    VNT *col_frequencies, *row_frequencies;
    MemoryAPI::allocate_array(&col_frequencies, _size);
    MemoryAPI::allocate_array(&row_frequencies, _size);

    for(VNT i = 0; i < size; i++)
    {
        col_frequencies[i] = 0;
        row_frequencies[i] = 0;
    }

    for(ENT i = 0; i < _nnz; i++) // TODO can be get from CSC / vise versa
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        col_frequencies[col]++;
        row_frequencies[row]++;
    }

    for(ENT i = 0; i < _nnz; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];
        tmp_col_ids[row].push_back(col);
        tmp_vals[row].push_back(val);
    }

    VNT *col_conversion_indexes, *row_conversion_indexes, *col_backward_conversion;
    MemoryAPI::allocate_array(&col_conversion_indexes, _size);
    MemoryAPI::allocate_array(&row_conversion_indexes, _size);
    MemoryAPI::allocate_array(&col_backward_conversion, _size);

    for(VNT i = 0; i < _size; i++) {
        col_conversion_indexes[i] = i;
        row_conversion_indexes[i] = i;
    }

    std::sort(col_conversion_indexes, col_conversion_indexes + _size,
              [col_frequencies](int index1, int index2)
              {
                  return col_frequencies[index1] > col_frequencies[index2];
              });

    std::sort(row_conversion_indexes, row_conversion_indexes + _size,
              [row_frequencies](int index1, int index2)
              {
                  return row_frequencies[index1] > row_frequencies[index2];
              });

    for(VNT i = 0; i < _size; i++)
    {
        col_backward_conversion[col_conversion_indexes[i]] = i;
    }

    resize(_size, _nnz, _target_socket);

    ENT cur_pos = 0;
    for(VNT row = 0; row < _size; row++)
    {
        VNT old_row = row_conversion_indexes[row];
        row_ptr[row] = cur_pos;
        row_ptr[row + 1] = cur_pos + tmp_col_ids[old_row].size();
        for(ENT j = 0; j < tmp_col_ids[old_row].size(); j++)
        {
            VNT new_col_id = col_backward_conversion[tmp_col_ids[old_row][j]];
            col_ids[cur_pos + j] = new_col_id;
            vals[cur_pos + j] = tmp_vals[old_row][j];
        }
        cur_pos += tmp_col_ids[old_row].size();
    }

    MemoryAPI::free_array(col_frequencies);
    MemoryAPI::free_array(row_frequencies);
    MemoryAPI::free_array(col_conversion_indexes);
    MemoryAPI::free_array(row_conversion_indexes);
    MemoryAPI::free_array(col_backward_conversion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSortCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _target_socket)
{
    construct_csr(_row_ids, _col_ids, _vals, _size, _nnz, _target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
