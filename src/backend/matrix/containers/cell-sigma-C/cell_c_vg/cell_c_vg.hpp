#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::CSRVertexGroupCellC()
{
    size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CSRVertexGroupCellC::id_in_range(VNT _src_id, VNT _nz_count)
{
    if ((_nz_count >= min_nz) && (_nz_count < max_nz))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::print()
{
    cout << "vertex group info: ";
    for (VNT i = 0; i < size; i++)
        cout << vertex_ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::~CSRVertexGroupCellC()
{
    MemoryAPI::free_array(vertex_ids);
    MemoryAPI::free_array(vector_group_ptrs);
    MemoryAPI::free_array(vector_group_sizes);
    MemoryAPI::free_array(vector_group_adjacent_ids);
    MemoryAPI::free_array(old_edge_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroupCellC::build(MatrixCellSigmaC<T> *_matrix, VNT _bottom, VNT _top)
{
    VNT matrix_size = _matrix->size;

    VNT local_group_size = 0;
    ENT local_group_total_nz = 0;

    min_nz = _bottom;
    max_nz = _top;

    // compute number of vertices and edges in vertex group
    for(VNT src_id = 0; src_id < matrix_size; src_id++)
    {
        VNT nz_count = _matrix->get_nz_count(src_id);
        if((nz_count >= _bottom) && (nz_count < _top))
        {
            local_group_total_nz += nz_count;
            local_group_size++;
        }
    }

    size = local_group_size;
    vector_segments_count = (size - 1) / VECTOR_LENGTH + 1;

    if(size == 0)
    {
        vector_segments_count = 0;
        edges_count_in_ve = 0;
        MemoryAPI::allocate_array(&vertex_ids, 1);
        MemoryAPI::allocate_array(&vector_group_ptrs, 1);
        MemoryAPI::allocate_array(&vector_group_sizes, 1);
        MemoryAPI::allocate_array(&vector_group_adjacent_ids, 1);
        MemoryAPI::allocate_array(&old_edge_indexes, 1);
    }
    else
    {
        MemoryAPI::allocate_array(&this->vertex_ids, size);

        // generate list of vertex group ids
        VNT vertex_pos = 0;
        for(VNT src_id = 0; src_id < matrix_size; src_id++)
        {
            VNT nz_count = _matrix->get_nz_count(src_id);
            if((nz_count >= _bottom) && (nz_count < _top))
            {
                this->vertex_ids[vertex_pos] = src_id;
                vertex_pos++;
            }
        }

        edges_count_in_ve = 0;
        for(VNT cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            VNT vec_start = cur_vector_segment * VECTOR_LENGTH;
            VNT cur_max_nz_count = 0;
            for(VNT i = 0; i < VECTOR_LENGTH; i++)
            {
                VNT vertex_pos = vec_start + i;
                if(vertex_pos < size)
                {
                    VNT src_id = this->vertex_ids[vertex_pos];
                    VNT nz_count = _matrix->get_nz_count(src_id);
                    if(cur_max_nz_count < nz_count)
                        cur_max_nz_count = nz_count;
                }
            }
            edges_count_in_ve += cur_max_nz_count * VECTOR_LENGTH;
        }
        MemoryAPI::allocate_array(&vector_group_ptrs, vector_segments_count);
        MemoryAPI::allocate_array(&vector_group_sizes, vector_segments_count);
        MemoryAPI::allocate_array(&vector_group_adjacent_ids, edges_count_in_ve + VECTOR_LENGTH);
        MemoryAPI::allocate_array(&old_edge_indexes, edges_count_in_ve + VECTOR_LENGTH);

        ENT current_edge = 0;
        for(VNT cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            VNT vec_start = cur_vector_segment * VECTOR_LENGTH;
            VNT cur_max_nz_count = 0;
            for(VNT i = 0; i < VECTOR_LENGTH; i++)
            {
                VNT vertex_pos = vec_start + i;
                if(vertex_pos < size)
                {
                    VNT src_id = this->vertex_ids[vertex_pos];
                    VNT nz_count = _matrix->get_nz_count(src_id);
                    if(cur_max_nz_count < nz_count)
                        cur_max_nz_count = nz_count;
                }
            }

            vector_group_ptrs[cur_vector_segment] = current_edge;
            vector_group_sizes[cur_vector_segment] = cur_max_nz_count;

            for(VNT edge_pos = 0; edge_pos < cur_max_nz_count; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vector
                for(VNT i = 0; i < VECTOR_LENGTH; i++)
                {
                    VNT vertex_pos = vec_start + i;
                    if(vertex_pos < size)
                    {
                        VNT src_id = this->vertex_ids[vertex_pos];
                        VNT nz_count = _matrix->get_nz_count(src_id);
                        if((vertex_pos < size) && (edge_pos < nz_count))
                        {
                            vector_group_adjacent_ids[current_edge + i] = _matrix->get_edge_dst(src_id, edge_pos);
                            old_edge_indexes[current_edge + i] = _matrix->get_edges_array_index(src_id, edge_pos);
                        }
                        else
                        {
                            vector_group_adjacent_ids[current_edge + i] = -1;
                            old_edge_indexes[current_edge + i] = -1;
                        }
                    }
                }
                current_edge += VECTOR_LENGTH;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

