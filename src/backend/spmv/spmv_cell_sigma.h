/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV_load_balanced(MatrixCellSigmaC<T> &_matrix,
                        DenseVector<T> &_x,
                        DenseVector<T> &_y)
{
    for(int vg = 0; vg < _matrix.vertex_groups_num; vg++)
    {
        VNT group_size = _matrix.vertex_groups[vg].size;
        #pragma omp parallel for schedule(static)
        for(VNT i = 0; i < group_size; i++)
        {
            VNT row = _matrix.vertex_groups[vg].ids[i];
            for(ENT j = _matrix.row_ptr[i]; j < _matrix.row_ptr[i + 1]; j++)
            {
                _y.vals[i] += _matrix.vals[j] * _x.vals[_matrix.col_ids[j]];
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
    SpMV_load_balanced(_matrix, _x, _y);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
