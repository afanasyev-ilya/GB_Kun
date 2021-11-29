/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV_load_balanced(MatrixCellSigmaC<T> &_matrix,
                        DenseVector<T> &_x,
                        DenseVector<T> &_y)
{
    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix.vertex_groups_num; vg++)
        {
            VNT group_size = _matrix.vertex_groups[vg].size;
            #pragma omp for schedule(static)
            for(VNT i = 0; i < group_size; i++)
            {
                VNT row = _matrix.vertex_groups[vg].ids[i];
                for(ENT j = _matrix.row_ptr[row]; j < _matrix.row_ptr[row + 1]; j++)
                {
                    _y.vals[row] += _matrix.vals[j] * _x.vals[_matrix.col_ids[j]];
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV_vector(MatrixCellSigmaC<T> &_matrix,
                 DenseVector<T> &_x,
                 DenseVector<T> &_y)
{
    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix.cell_c_start_group; vg++)
        {
            #pragma omp for schedule(static)
            for(VNT i = 0; i < _matrix.vertex_groups[vg].size; i++)
            {
                VNT row = _matrix.vertex_groups[vg].ids[i];
                for(ENT j = _matrix.row_ptr[row]; j < _matrix.row_ptr[row + 1]; j++)
                {
                    _y.vals[row] += _matrix.vals[j] * _x.vals[_matrix.col_ids[j]];
                }
            }
        }

        for(int vg = 0; vg < _matrix.cell_c_vertex_groups_num; vg++)
        {
            VNT vector_segments_count = _matrix.cell_c_vertex_groups[vg].vector_segments_count;

            #pragma omp for schedule(static)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                VNT segment_first_vertex = cur_vector_segment * VECTOR_LENGTH;

                ENT segment_edges_start = _matrix.cell_c_vertex_groups[vg].vector_group_ptrs[cur_vector_segment];
                VNT segment_connections_count = _matrix.cell_c_vertex_groups[vg].vector_group_sizes[cur_vector_segment];

                for(ENT edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    for (VNT i = 0; i < VECTOR_LENGTH; i++)
                    {
                        VNT pos = segment_first_vertex + i;
                        VNT row_id = 0;
                        VNT size = _matrix.cell_c_vertex_groups[vg].size;
                        if(pos < size)
                        {
                            row_id = _matrix.cell_c_vertex_groups[vg].row_ids[pos];
                        }

                        if(pos < size)
                        {
                            const VNT vector_index = i;
                            const ENT internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                            const VNT local_edge_pos = edge_pos;
                            const VNT col_id = _matrix.cell_c_vertex_groups[vg].vector_group_col_ids[internal_edge_pos];
                            if(col_id != -1)
                            {
                                const T val = _matrix.cell_c_vertex_groups[vg].vector_group_vals[internal_edge_pos];
                                _y.vals[row_id] += val * _x.vals[col_id];
                            }
                        }
                    }
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixCellSigmaC<T> &_matrix,
          DenseVector<T> &_x,
          DenseVector<T> &_y)
{
    //SpMV_load_balanced(_matrix, _x, _y);
    SpMV_vector(_matrix, _x, _y);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
