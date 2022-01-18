/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::prepare_vg_lists(int _target_socket)
{
    // set thresholds
    ENT step = 16;
    ENT first = 4;
    vertex_groups[0].set_thresholds(0, first);
    for(int i = 1; i < (vg_num - 1); i++)
    {
        vertex_groups[i].set_thresholds(first, first*step);
        first *= step;
    }
    vertex_groups[vg_num - 1].set_thresholds(first, INT_MAX);

    // push back vertices to each group using O|V| work
    for(VNT row = 0; row < size; row++)
    {
        ENT connections_count = row_ptr[row + 1] - row_ptr[row];
        for(int vg = 0; vg < vg_num; vg++)
            if(vertex_groups[vg].in_range(connections_count))
                vertex_groups[vg].push_back(row);
    }

    // create optimized representation for each socket
    for(int i = 0; i < vg_num; i++)
    {
        vertex_groups[i].finalize_creation(_target_socket);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _target_socket)
{
    vector<vector<pair<VNT, T>>> tmp_csr;

    edges_list_to_vector_of_vectors(_row_ids, _col_ids, _vals, _size, _nnz, tmp_csr);

    resize(_size, _nnz, _target_socket);

    vector_of_vectors_to_csr(tmp_csr, row_ptr, col_ids, vals);

    prepare_vg_lists(_target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(vector<vector<pair<VNT, T>>> &_tmp_csr_matrix, int _target_socket)
{
    cout << "im here" << endl;
    throw "error";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
