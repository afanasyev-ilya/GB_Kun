/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixVectGroupCSR<T>::construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz)
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

template<typename T>
void MatrixVectGroupCSR<T>::create_vertex_groups()
{
    vertex_groups_num = 6;
    vertex_groups = new CSRVertexGroup<T>[vertex_groups_num];
    vertex_groups[0].build(this, 256, 2147483647);
    vertex_groups[1].build(this, 128, 256);
    vertex_groups[2].build(this, 64, 128);
    vertex_groups[3].build(this, 32, 64);
    vertex_groups[4].build(this, 16, 32);
    vertex_groups[5].build(this, 0, 16);

    if(VECTOR_LENGTH == 32)
        cell_c_start_group = 1;
    else
        throw "Error: incorrect VECTOR_LENGTH in MatrixVectGroupCSR<T>::create_vertex_groups";

    cell_c_vertex_groups_num = 6;
    cell_c_vertex_groups = new CSRVertexGroupSellC<T>[cell_c_vertex_groups_num];
    cell_c_vertex_groups[0].build(this, 256, 512);
    cell_c_vertex_groups[1].build(this, 128, 256);
    cell_c_vertex_groups[2].build(this, 64, 128);
    cell_c_vertex_groups[3].build(this, 32, 64);
    cell_c_vertex_groups[4].build(this, 16, 32);
    cell_c_vertex_groups[5].build(this, 0, 16);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixVectGroupCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    resize(_size, _nz);
    construct_unsorted_csr(_row_ids, _col_ids, _vals, _size, _nz);
    create_vertex_groups();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
