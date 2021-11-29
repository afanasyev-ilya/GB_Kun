#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend {

template <typename T>
void SpMV(const MatrixSegmentedCSR<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    int cores_num = omp_get_max_threads();
    #pragma omp parallel
    {
        for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
        {
            SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
            T *buffer = (T*)segment->vertex_buffer;

            #pragma omp for schedule(static)
            for(VNT i = 0; i < segment->size; i++)
                buffer[i] = 0;

            #pragma omp for schedule(guided, 1024)
            for(VNT i = 0; i < segment->size; i++)
            {
                for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
                {
                    buffer[i] += segment->vals[j] * x_vals[segment->col_ids[j]];
                }
            }
        }
    }

    if(_matrix->size > pow(2.0, 22)) // cache aware merge
    {
        int outer_threads = min(_matrix->merge_blocks_number, cores_num);
        int inner_threads = cores_num/outer_threads;
        #pragma omp parallel num_threads(outer_threads)
        {
            #pragma omp for schedule(static)
            for(VNT cur_block = 0; cur_block < _matrix->merge_blocks_number; cur_block++)
            {
                for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
                {
                    SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
                    T *buffer = (T*)segment->vertex_buffer;
                    VNT *conversion_indexes = segment->conversion_to_full;

                    VNT block_start = segment->block_starts[cur_block];
                    VNT block_end = segment->block_ends[cur_block];

                    #pragma omp parallel for num_threads(inner_threads)
                    for(VNT i = block_start; i < block_end; i++)
                    {
                        y_vals[conversion_indexes[i]] += buffer[i];
                    }
                }
            }
        };
        double t4 = omp_get_wtime();
    }
    else
    {
        for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
        {
            SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
            T *buffer = (T*)segment->vertex_buffer;
            VNT *conversion_indexes = segment->conversion_to_full;

            #pragma omp parallel for schedule(static)
            for(VNT i = 0; i < segment->size; i++)
            {
                y_vals[conversion_indexes[i]] += buffer[i];
            }
        }
    }

    //cout << "compare: " << (t2 - t1)*1000 << "(edge proc) vs " << (t4 - t3)*1000 << "(cache-aware) vs " << (t6 - t5)*1000 << "(usual merge)" << endl;
}

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

