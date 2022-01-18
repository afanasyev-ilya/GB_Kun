#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void edges_list_to_vector_of_vectors(const VNT *_row_ids,
                                     const VNT *_col_ids,
                                     const T *_vals,
                                     VNT _size,
                                     ENT _nnz,
                                     vector<vector<pair<VNT, T>>> &_result)
{
    _result.resize(_size);

    for(ENT i = 0; i < _nnz; i++)
    {
        VNT row = _row_ids[i];
        VNT col = _col_ids[i];
        T val = _vals[i];
        _result[row].push_back(make_pair(col, val));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void vector_of_vectors_to_csr(vector<vector<pair<VNT, T>>> &_vec,
                              ENT *_row_ptr,
                              VNT *_col_ids,
                              T *_vals)
{
    ENT cur_pos = 0;
    for(VNT i = 0; i < _vec.size(); i++)
    {
        _row_ptr[i] = cur_pos;
        _row_ptr[i + 1] = cur_pos + _vec[i].size();
        for(ENT j = _row_ptr[i]; j < _row_ptr[i + 1]; j++)
        {
            _col_ids[j] = _vec[i][j - _row_ptr[i]].first;
            _vals[j] = _vec[i][j - _row_ptr[i]].second;
        }
        cur_pos += _vec[i].size();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
