#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend {

template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
void SpMV(const MatrixSegmentedCSR<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          BinaryOpTAccum _accum,
          SemiringT op,
          Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    Y *shared_vector = (Y*)_workspace->get_first_socket_vector();

    double t1 = omp_get_wtime();
    int cores_num = omp_get_max_threads();
    #pragma omp parallel // parallelism within different segments
    {
        for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
        {
            SubgraphSegment<A> *segment = &(_matrix->subgraphs[seg_id]);
            Y *buffer = (Y*)segment->vertex_buffer;

            if(segment->static_ok_to_use)
            {
                #pragma omp for nowait schedule(static, 32)
                for(VNT i = 0; i < segment->size; i++)
                {
                    Y res = identity_val;
                    for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
                    {
                        res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
                    }
                    buffer[i] = res;
                }
            }
            else
            {
                for(int vg = 0; vg < segment->vg_num; vg++)
                {
                    const VNT *vertices = segment->vertex_groups[vg].get_data();
                    VNT vertex_group_size = segment->vertex_groups[vg].get_size();

                    #pragma omp for nowait schedule(static)
                    for(VNT idx = 0; idx < vertex_group_size; idx++)
                    {
                        VNT row = vertices[idx];
                        Y res = identity_val;
                        for(ENT j = segment->row_ptr[row]; j < segment->row_ptr[row + 1]; j++)
                        {
                            res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
                        }
                        buffer[row] = res;
                    }
                }
            }
        }
    }

    /*for(int s = 0; s < _matrix->num_segments; s++)
    {
        int seg_id = s;

        SubgraphSegment<A> *segment = &(_matrix->subgraphs[seg_id]);
        Y *buffer = (Y *) segment->vertex_buffer;

        double t1_in = omp_get_wtime();
        #pragma omp parallel for schedule(static, 32)
        for(VNT i = 0; i < segment->size; i++)
        {
            Y res = identity_val;
            for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
            {
                res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
            }
            buffer[i] = res;
        }
        double t2_in = omp_get_wtime();
        cout << "seg  " << seg_id << " BW: " << segment->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2_in - t1_in)*1e9)
        << " GB/s, avg_deg = " << ((double)segment->nnz)/segment->size << ", " <<
           100.0*(double)segment->nnz/_matrix->nnz << "% of nnz" << endl;
    }*/

    /*#pragma omp parallel  // testing number of processed edges and manual static parallelism
    {
        int tid = omp_get_thread_num();
        int seg_id = _matrix->small_segments[tid / 4];
        int inner_tid = tid % 4;

        ENT proc_edges = 0;

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
                proc_edges += segment->row_ptr[i + 1] - segment->row_ptr[i];
            }
        }

        seg_id = _matrix->small_segments[_matrix->largest_segment];
        segment = &(_matrix->subgraphs[seg_id]);
        buffer = (T*)segment->vertex_buffer;
        for(VNT i = 0; i < segment->size; i++)
            buffer[i] = 0;

        #pragma omp for schedule(guided, 128)
        for(VNT i = 0; i < segment->size; i++)
        {
            T res = identity_val;
            for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
            {
                res = add_op(res, mul_op(segment->vals[j], x_vals[segment->col_ids[j]]));
            }
            buffer[i] = res;
            proc_edges += segment->row_ptr[i + 1] - segment->row_ptr[i];
        }

        #pragma omp critical
        {
            cout << proc_edges << " / " << _matrix->get_nnz() << ", " << 100.0 * (double)proc_edges/_matrix->get_nnz() << "%" << endl;
        }
    }*/

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
    cout << "inner (seg) time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "inner (seg) BW: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    double t3 = omp_get_wtime();
    if(_matrix->merge_blocks_number >= 4*cores_num)
    {
        cout << "using cache-aware merge" << endl;
        #pragma omp parallel // cache aware scatter merge
        {
            #pragma omp for
            for(VNT i = 0; i < _matrix->size; i++)
            {
                shared_vector[i] = identity_val;
            }

            #pragma omp for schedule(static, 32)
            for(VNT cur_block = 0; cur_block < _matrix->merge_blocks_number; cur_block++)
            {
                for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
                {
                    SubgraphSegment<A> *segment = &(_matrix->subgraphs[seg_id]);
                    Y *buffer = (Y*)segment->vertex_buffer;
                    VNT *conversion_indexes = segment->conversion_to_full;

                    VNT block_start = segment->block_starts[cur_block];
                    VNT block_end = segment->block_ends[cur_block];

                    for(VNT i = block_start; i < block_end; i++)
                    {
                        shared_vector[conversion_indexes[i]] = add_op(shared_vector[conversion_indexes[i]], buffer[i]);
                    }
                }
            }

            #pragma omp for
            for(VNT i = 0; i < _matrix->size; i++)
            {
                y_vals[i] = _accum(y_vals[i], shared_vector[i]);
            }
        }
    }
    else
    {
        cout << "using simple merge" << endl;
        #pragma omp parallel // non cache aware scatter merge
        {
            #pragma omp for
            for(VNT i = 0; i < _matrix->size; i++)
            {
                shared_vector[i] = identity_val;
            }

            for(int seg_id = 0; seg_id < _matrix->num_segments; seg_id++)
            {
                SubgraphSegment<A> *segment = &(_matrix->subgraphs[seg_id]);
                Y *buffer = (Y*)segment->vertex_buffer;
                VNT *conversion_indexes = segment->conversion_to_full;

                #pragma omp for schedule(static, 32)
                for(VNT i = 0; i < segment->size; i++)
                {
                    shared_vector[conversion_indexes[i]] = add_op(shared_vector[conversion_indexes[i]], buffer[i]);
                }
            }

            #pragma omp for
            for(VNT i = 0; i < _matrix->size; i++)
            {
                y_vals[i] = _accum(y_vals[i], shared_vector[i]);
            }
        }
    }

    double t4 = omp_get_wtime();
    cout << "merge time: " << (t4 - t3)*1000 << " ms" << endl;
    cout << "wall spmv time: " << (t4 - t1)*1000 << " ms" << endl;
    cout << "bw: " << _matrix->nnz * (2.0*sizeof(X) + sizeof(Index)) / ((t4 - t1)*1e9) << " GB/s" << endl << endl;

    //cout << "compare: " << (t2 - t1)*1000 << "(edge proc) vs " << (t4 - t3)*1000 << "(cache-aware) vs " << (t6 - t5)*1000 << "(usual merge)" << endl;
}

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

