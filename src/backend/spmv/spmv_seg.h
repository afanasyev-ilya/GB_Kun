#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend {

template <typename T, typename  SemiringT>
void SpMV(const MatrixSegmentedCSR<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    double t1 = omp_get_wtime();

    int cores_num = omp_get_max_threads();
    #pragma omp parallel
    {
        for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
        {
            SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
            T *buffer = (T*)segment->vertex_buffer;

            #pragma omp for nowait schedule(guided, 1)
            for(VNT i = 0; i < segment->size; i++)
            {
                T res = identity_val;
                for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
                {
                    res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
                }
                buffer[i] = res;
            }
        }
    }
    double t2 = omp_get_wtime();
    cout << "inner time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "inner BW: " << _matrix->nz * (2.0*sizeof(T) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    if(_matrix->size > pow(2.0, 22)) // cache aware merge
    {
        int outer_threads = std::min((int)_matrix->merge_blocks_number, cores_num);
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
                        y_vals[conversion_indexes[i]] = buffer[i];
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

