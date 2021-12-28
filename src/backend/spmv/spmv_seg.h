#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend {

template <typename T, typename  SemiringT>
void SpMV(const MatrixSegmentedCSR<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    _matrix->print();
    _x->print();
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    double t1 = omp_get_wtime();

    int cores_num = omp_get_max_threads();
    /*#pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
        {
            SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
            T *buffer = (T*)segment->vertex_buffer;

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
    }*/
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int seg_id = tid / 4;
        int inner_tid = tid % 4;

        SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
        T *buffer = (T*)segment->vertex_buffer;

        VNT work_size = (segment->size - 1)/4 + 1;
        for(VNT i = inner_tid*work_size; i < (inner_tid+1)*work_size; i++)
            if(i < segment->size)
                buffer[i] = 0;

        for(VNT i = inner_tid*work_size; i < (inner_tid+1)*work_size; i++)
        {
            if(i < segment->size)
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
    /*#pragma omp parallel num_threads(12)
    {
        int seg_id = omp_get_thread_num();
        SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
        T *buffer = (T*)segment->vertex_buffer;

        #pragma omp parallel num_threads(4)
        {
            #pragma omp for schedule(static)
            for(VNT i = 0; i < segment->size; i++)
                buffer[i] = 0;

            #pragma omp for schedule(static)
            for(VNT i = 0; i < segment->size; i ++)
            {
                T res = identity_val;
                for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
                {
                    res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
                }
                buffer[i] = res;
            }
        }
    }*/
    double t2 = omp_get_wtime();
    cout << "inner 5 time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "inner 5 BW: " << _matrix->nnz * (2.0*sizeof(T) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    if(_matrix->size > pow(2.0, 23)) // cache aware merge
    {
        int outer_threads = std::min((int)_matrix->merge_blocks_number, cores_num);
        int inner_threads = cores_num/outer_threads;
        #pragma omp parallel
        {
            for(VNT cur_block = 0; cur_block < _matrix->merge_blocks_number; cur_block++)
            {
                for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
                {
                    SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
                    T *buffer = (T*)segment->vertex_buffer;
                    VNT *conversion_indexes = segment->conversion_to_full;

                    VNT block_start = segment->block_starts[cur_block];
                    VNT block_end = segment->block_ends[cur_block];

                    #pragma omp for schedule(static)
                    for(VNT i = block_start; i < block_end; i++)
                    {
                        y_vals[conversion_indexes[i]] += buffer[i];
                    }
                }
            }
        };
    }
    else
    {
        #pragma omp parallel
        {
            for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
            {
                SubgraphSegment<T> *segment = &(_matrix->subgraphs[seg_id]);
                T *buffer = (T*)segment->vertex_buffer;
                VNT *conversion_indexes = segment->conversion_to_full;

                #pragma omp for schedule(static)
                for(VNT i = 0; i < segment->size; i++)
                {
                    y_vals[conversion_indexes[i]] += buffer[i];
                }
            }
        }

    }
    t2 = omp_get_wtime();
    cout << "merge time: " << (t2 - t1)*1000 << " ms" << endl;

    //cout << "compare: " << (t2 - t1)*1000 << "(edge proc) vs " << (t4 - t3)*1000 << "(cache-aware) vs " << (t6 - t5)*1000 << "(usual merge)" << endl;
}

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

