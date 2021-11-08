/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::construct_unsorted_csr(const VNT *_row_ids,
                                          const VNT *_col_ids,
                                          T *_vals,
                                          VNT _size,
                                          ENT _non_zeroes_num)
{
    vector<vector<VNT>> tmp_col_ids(_size);
    vector<vector<T>> tmp_vals(_size);

    for(ENT i = 0; i < _non_zeroes_num; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];
        tmp_col_ids[row].push_back(col);
        tmp_vals[row].push_back(val);
    }

    resize(_size, _non_zeroes_num);

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
void MatrixCSR<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _non_zeroes_num)
{
    resize(_size, _non_zeroes_num);
    construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _non_zeroes_num);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
