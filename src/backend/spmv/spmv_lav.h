#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend {

template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
void SpMV(const MatrixLAV<A> *_matrix,
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
    Y *prefetched_vector = (Y*)_workspace->get_prefetched_vector();

    #pragma omp parallel for
    for(VNT i = 0; i < _matrix->nrows; i++)
        prefetched_vector[i] = x_vals[_matrix->col_new_to_old[i]];

    VNT dense_segments_num = _matrix->dense_segments_num;
    VNT num_rows = _matrix->nrows;
    double t1, t2;
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int cur_seg = 0;
        const LAVSegment<A> *segment_data = &(_matrix->dense_segments[cur_seg]);
        const VNT *row_ids = segment_data->vertex_list.get_data();
        const VNT nnz_num_rows = segment_data->vertex_list.get_size();

        #pragma omp for schedule(static)
        for(VNT idx = 0; idx < _matrix->nrows; idx++)
        {
            shared_vector[idx] = identity_val;
        }

        /*VNT min_col = segment_data->min_col_id;
        VNT max_col = segment_data->max_col_id;
        #pragma omp for schedule(static)
        for(VNT i = min_col; i <= max_col; i++)
        {
            prefetched_vector[i - min_col] = x_vals[i];
        }*/

        ENT proc_edges = 0;
        for(int vg = 0; vg < segment_data->vg_num; vg++)
        {
            const VNT *vertices = segment_data->vertex_groups[vg].get_data();
            VNT vertex_group_size = segment_data->vertex_groups[vg].get_size();

            #pragma omp for nowait schedule(static)
            for(VNT idx = 0; idx < vertex_group_size; idx++)
            {
                VNT row = vertices[idx];
                Y res = identity_val;

                for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
                {
                    VNT col = segment_data->col_ids[j];
                    Y mat_val = segment_data->vals[j];
                    X x_val = prefetched_vector[col];
                    res = add_op(res, mul_op(mat_val, x_val));
                }
                shared_vector[row] = add_op(shared_vector[row], res);
            }
        }

        /*#pragma omp for schedule(guided, 256)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            Y res = identity_val;
            for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
            {
                VNT col = segment_data->col_ids[j];
                Y val = segment_data->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            shared_vector[row] = add_op(shared_vector[row], res);
        }*/
    }
    t2 = omp_get_wtime();
    cout << "largest BW: " << _matrix->dense_segments[0].nnz * (2*sizeof(A) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        for(VNT cur_seg = 1; cur_seg < dense_segments_num; cur_seg++)
        {
            const LAVSegment<A> *segment_data = &(_matrix->dense_segments[cur_seg]);
            const VNT *row_ids = segment_data->vertex_list.get_data();
            const VNT nnz_num_rows = segment_data->vertex_list.get_size();

            #pragma omp for schedule(guided, 1)
            for(VNT idx = 0; idx < nnz_num_rows; idx++)
            {
                VNT row = row_ids[idx];
                Y res = identity_val;
                for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
                {
                    VNT col = segment_data->col_ids[j];
                    Y val = segment_data->vals[j];
                    res = add_op(res, mul_op(val, prefetched_vector[col]));
                }
                shared_vector[row] = add_op(shared_vector[row], res);
            }
        }
    }
    ENT mid_sum = 0;
    for(VNT cur_seg = 1; cur_seg < dense_segments_num; cur_seg++)
    {
        mid_sum += _matrix->dense_segments[cur_seg].nnz;
    }
    t2 = omp_get_wtime();
    cout << "mid BW: " << mid_sum * (2*sizeof(A) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        const LAVSegment<A> *segment_data = &(_matrix->sparse_segment);
        const VNT *row_ids = segment_data->vertex_list.get_data();
        const VNT nnz_num_rows = segment_data->vertex_list.get_size();

        #pragma omp for schedule(guided, 1)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            Y res = identity_val;
            for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
            {
                VNT col = segment_data->col_ids[j];
                Y val = segment_data->vals[j];
                res = add_op(res, mul_op(val, prefetched_vector[col]));
            }
            shared_vector[row] = add_op(shared_vector[row], res);
        }

        #pragma omp for schedule(static)
        for(VNT row = 0; row < _matrix->nrows; row++)
        {
            y_vals[row] = _accum(y_vals[row], shared_vector[row]);
        }
    }
    t2 = omp_get_wtime();
    cout << "sparse BW: " << _matrix->sparse_segment.nnz * (2*sizeof(A) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl << endl;

    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

