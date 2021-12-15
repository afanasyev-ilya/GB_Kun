#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CSRVertexGroupSellC<T>::CSRVertexGroupSellC()
{
    size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool CSRVertexGroupSellC<T>::id_in_range(VNT _src_id, VNT _nnz_count)
{
    if ((_nnz_count >= min_nnz) && (_nnz_count < max_nnz))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroupSellC<T>::print()
{
    cout << "vertex group info: ";
    for (VNT i = 0; i < size; i++)
        cout << row_ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CSRVertexGroupSellC<T>::~CSRVertexGroupSellC()
{
    MemoryAPI::free_array(row_ids);
    MemoryAPI::free_array(vector_group_ptrs);
    MemoryAPI::free_array(vector_group_sizes);
    MemoryAPI::free_array(vector_group_col_ids);
    MemoryAPI::free_array(vector_group_vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroupSellC<T>::build(MatrixVectGroupCSR<T> *_matrix, VNT _bottom, VNT _top)
{
    VNT matrix_size = _matrix->size;

    VNT local_group_size = 0;
    ENT local_group_total_nnz = 0;

    min_nnz = _bottom;
    max_nnz = _top;

    // compute number of vertices and edges in vertex group
    for(VNT src_id = 0; src_id < matrix_size; src_id++)
    {
        VNT nnz_count = _matrix->get_nnz_in_row(src_id);
        if((nnz_count >= _bottom) && (nnz_count < _top))
        {
            local_group_total_nnz += nnz_count;
            local_group_size++;
        }
    }

    size = local_group_size;
    vector_segments_count = (size - 1) / VECTOR_LENGTH + 1;

    if(size == 0)
    {
        vector_segments_count = 0;
        edges_count_in_ve = 0;
        MemoryAPI::allocate_array(&row_ids, 1);
        MemoryAPI::allocate_array(&vector_group_ptrs, 1);
        MemoryAPI::allocate_array(&vector_group_sizes, 1);
        MemoryAPI::allocate_array(&vector_group_col_ids, 1);
        MemoryAPI::allocate_array(&vector_group_vals, 1);
    }
    else
    {
        MemoryAPI::allocate_array(&this->row_ids, size);

        // generate list of vertex group ids
        VNT vertex_pos = 0;
        for(VNT src_id = 0; src_id < matrix_size; src_id++)
        {
            VNT nnz_count = _matrix->get_nnz_in_row(src_id);
            if((nnz_count >= _bottom) && (nnz_count < _top))
            {
                this->row_ids[vertex_pos] = src_id;
                vertex_pos++;
            }
        }

        edges_count_in_ve = 0;
        for(VNT cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            VNT vec_start = cur_vector_segment * VECTOR_LENGTH;
            VNT cur_max_nnz_count = 0;
            for(VNT i = 0; i < VECTOR_LENGTH; i++)
            {
                VNT vertex_pos = vec_start + i;
                if(vertex_pos < size)
                {
                    VNT src_id = this->row_ids[vertex_pos];
                    VNT nnz_count = _matrix->get_nnz_in_row(src_id);
                    if(cur_max_nnz_count < nnz_count)
                        cur_max_nnz_count = nnz_count;
                }
            }
            edges_count_in_ve += cur_max_nnz_count * VECTOR_LENGTH;
        }
        MemoryAPI::allocate_array(&vector_group_ptrs, vector_segments_count);
        MemoryAPI::allocate_array(&vector_group_sizes, vector_segments_count);
        MemoryAPI::allocate_array(&vector_group_col_ids, edges_count_in_ve + VECTOR_LENGTH);
        MemoryAPI::allocate_array(&vector_group_vals, edges_count_in_ve + VECTOR_LENGTH);

        ENT current_edge = 0;
        for(VNT cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            VNT vec_start = cur_vector_segment * VECTOR_LENGTH;
            VNT cur_max_nnz_count = 0;
            for(VNT i = 0; i < VECTOR_LENGTH; i++)
            {
                VNT vertex_pos = vec_start + i;
                if(vertex_pos < size)
                {
                    VNT src_id = this->row_ids[vertex_pos];
                    VNT nnz_count = _matrix->get_nnz_in_row(src_id);
                    if(cur_max_nnz_count < nnz_count)
                        cur_max_nnz_count = nnz_count;
                }
            }

            vector_group_ptrs[cur_vector_segment] = current_edge;
            vector_group_sizes[cur_vector_segment] = cur_max_nnz_count;

            for(VNT edge_pos = 0; edge_pos < cur_max_nnz_count; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vector
                for(VNT i = 0; i < VECTOR_LENGTH; i++)
                {
                    VNT vertex_pos = vec_start + i;
                    if(vertex_pos < size)
                    {
                        VNT src_id = this->row_ids[vertex_pos];
                        VNT nnz_count = _matrix->get_nnz_in_row(src_id);
                        if((vertex_pos < size) && (edge_pos < nnz_count))
                        {
                            VNT col_id = _matrix->col_ids[_matrix->row_ptr[src_id] + edge_pos];
                            T val = _matrix->vals[_matrix->row_ptr[src_id] + edge_pos];
                            vector_group_col_ids[current_edge + i] = col_id;
                            vector_group_vals[current_edge + i] = val;
                        }
                        else
                        {
                            vector_group_col_ids[current_edge + i] = 0;//-1;
                            vector_group_vals[current_edge + i] = 0;//-1;
                        }
                    }
                }
                current_edge += VECTOR_LENGTH;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

