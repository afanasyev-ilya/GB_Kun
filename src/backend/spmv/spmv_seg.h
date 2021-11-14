#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixSegmentedCSR<T> &_matrix, DenseVector<T> &_x, DenseVector<T> &_y)
{
    for(int seg_id = 0; seg_id < _matrix.num_segments; seg_id++)
    {
        SubgraphSegment<T> *segment = &(_matrix.subgraphs[seg_id]);
        T *buffer = (T*)segment->vertex_buffer;

        #pragma omp parallel for schedule(static)
        for(VNT i = 0; i < segment->size; i++)
        {
            buffer[i] = 0;
            for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
            {
                buffer[i] += segment->vals[j] * _x.vals[segment->col_ids[j]];
            }
        }
    }

    for(int seg_id = 0; seg_id < _matrix.num_segments; seg_id++)
    {
        SubgraphSegment<T> *segment = &(_matrix.subgraphs[seg_id]);
        T *buffer = (T*)segment->vertex_buffer;
        VNT *conversion_indexes = segment->conversion_to_full;

        #pragma omp parallel for schedule(static)
        for(VNT i = 0; i < segment->size; i++)
        {
            _y.vals[conversion_indexes[i]] += buffer[i];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

