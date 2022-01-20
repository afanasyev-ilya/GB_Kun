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

    /*int part_size = 2;

    while((part_size <= 8192) && (part_size < _size))
    {
        ENT cnt = 0;
        for(VNT x = 0; x < part_size; x++)
        {
            for(VNT y = 0; y < part_size; y++)
            {
                T val = get(x, y);
                if(val != 0)
                    cnt++;
            }
        }
        cout << cnt << " " << part_size*part_size << " | " << _nnz << endl;
        cout << "dense percent: " << (100.0*cnt) / (part_size * part_size) << " % for part = " << part_size << endl;
        cout << "perc of all: " << (100.0*cnt) / _nnz << " % of all edges" << endl;
        cout << sizeof(T)*part_size*part_size/1e9 << " GB is size of dense part" << endl << endl;
        part_size *= 2;
    }*/

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
