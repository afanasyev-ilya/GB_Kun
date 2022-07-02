/**
  @file format_conversions.h
  @author S.krymskiy
  @version Revision 1.1
  @date June 10, 2022
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * edges_list_to_vector_of_vectors function.
 * @brief converts list of edges to the adjacency list
 * @param _row_ids pointer to an array containing the row number of nonzero elements
 * @param _col_ids pointer to an array containing the column number of nonzero elements
 * @param _vals pointer an array containing values of nonzero elements
 * @param _size size of the resulting vector of vectors (number of vertexes)
 * @param _nnz number of nonzero elements
 * @param _result resulting vector of vectors
*/

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

/**
 * vector_of_vectors_to_csr function.
 * @brief converts an adjacency list to the CSR format
 * @param _tmp_mat vector of vectors containing all nonzero elements
 * @param _row_ids pointer to an array containing the row number of nonzero elements
 * @param _col_ids pointer to an array containing the column number of nonzero elements
 * @param _vals pointer an array containing values of nonzero elements
*/

template <typename T>
void vector_of_vectors_to_csr(vector<vector<pair<VNT, T>>> &_tmp_mat,
                              ENT *_row_ptr,
                              VNT *_col_ids,
                              T *_vals)
{
    ENT cur_pos = 0;
    for(VNT i = 0; i < _tmp_mat.size(); i++)
    {
        _row_ptr[i] = cur_pos;
        _row_ptr[i + 1] = cur_pos + _tmp_mat[i].size();
        cur_pos += _tmp_mat[i].size();
    }

    #pragma omp parallel for schedule(guided, 1024)
    for(VNT i = 0; i < _tmp_mat.size(); i++)
    {
        for(ENT j = _row_ptr[i]; j < _row_ptr[i + 1]; j++)
        {
            _col_ids[j] = _tmp_mat[i][j - _row_ptr[i]].first;
            _vals[j] = _tmp_mat[i][j - _row_ptr[i]].second;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * vector_of_maps_to_csr function.
 * @brief converts an adjacency list to the CSR format
 * @param _tmp_mat vector of maps containing all nonzero elements
 * @param _row_ids pointer to an array containing the row number of nonzero elements
 * @param _col_ids pointer to an array containing the column number of nonzero elements
 * @param _vals pointer an array containing values of nonzero elements
*/

template <typename T>
void vector_of_maps_to_csr(std::vector<std::unordered_map<VNT, T>> &_tmp_mat,
                           ENT *_row_ptr,
                           VNT *_col_ids,
                           T *_vals)
{
    ENT cur_pos = 0;
    for(VNT i = 0; i < _tmp_mat.size(); i++)
    {
        _row_ptr[i] = cur_pos;
        _row_ptr[i + 1] = cur_pos + _tmp_mat[i].size();
        cur_pos += _tmp_mat[i].size();
    }

    #pragma omp parallel for schedule(guided, 1024)
    for(VNT row = 0; row < _tmp_mat.size(); row++)
    {
        ENT j = _row_ptr[row];
        for(auto &[col_id, val]: _tmp_mat[row])
        {
            _col_ids[j] = col_id;
            _vals[j] = val;
            j++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * estimate_nnz_in_vector_of_vectors function.
 * @brief returns number of nonzero elements in an adjacency list
 * @param _tmp_mat vector of vectors containing all nonzero elements
 * @return number of nonzero elements
*/

template <typename T>
ENT estimate_nnz_in_vector_of_vectors(vector<vector<pair<VNT, T>>> &_tmp_mat)
{
    ENT nnz = 0;
    #pragma omp parallel for reduction(+: nnz)
    for(VNT i = 0; i < _tmp_mat.size(); i++)
    {
        nnz += _tmp_mat[i].size();
    }
    return nnz;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
