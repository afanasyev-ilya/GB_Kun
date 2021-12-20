#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend {

template <typename T, typename SemiringT>
void SpMV(const MatrixLAV<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    VNT dense_segments_num = _matrix->dense_segments_num;
    VNT num_rows = _matrix->size;
    double t1, t2;
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int cur_seg = 0;
        const LAVSegment<T> *segment_data = &(_matrix->dense_segments[cur_seg]);
        const VNT *row_ids = segment_data->vertex_list.ptr();
        const VNT nnz_num_rows = segment_data->vertex_list.size();

        ENT proc_elems = 0;

        #pragma omp for schedule(guided, 1024)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            T res = identity_val;
            for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
            {
                VNT col = segment_data->col_ids[j];
                T val = segment_data->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
                proc_elems++;
            }
            y_vals[row] = add_op(y_vals[row], res);
        }

        #pragma omp critical
        {
            cout << proc_elems << " / " << segment_data->nnz << ", " << 100.0*(double)proc_elems/segment_data->nnz <<  endl;
        }
    }
    t2 = omp_get_wtime();
    cout << "largest BW: " << _matrix->dense_segments[0].nnz * (2*sizeof(T) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        for(VNT cur_seg = 1; cur_seg < dense_segments_num; cur_seg++)
        {
            const LAVSegment<T> *segment_data = &(_matrix->dense_segments[cur_seg]);
            const VNT *row_ids = segment_data->vertex_list.ptr();
            const VNT nnz_num_rows = segment_data->vertex_list.size();

            #pragma omp for schedule(guided, 1)
            for(VNT idx = 0; idx < nnz_num_rows; idx++)
            {
                VNT row = row_ids[idx];
                T res = identity_val;
                for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
                {
                    VNT col = segment_data->col_ids[j];
                    T val = segment_data->vals[j];
                    res = add_op(res, mul_op(val, x_vals[col]));
                }
                y_vals[row] = add_op(y_vals[row], res);
            }
        }
    }
    ENT mid_sum = 0;
    for(VNT cur_seg = 1; cur_seg < dense_segments_num; cur_seg++)
    {
        mid_sum += _matrix->dense_segments[cur_seg].nnz;
    }
    t2 = omp_get_wtime();
    cout << "mid BW: " << mid_sum * (2*sizeof(T) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl;


    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        const LAVSegment<T> *segment_data = &(_matrix->sparse_segment);
        const VNT *row_ids = segment_data->vertex_list.ptr();
        const VNT nnz_num_rows = segment_data->vertex_list.size();

        #pragma omp for schedule(guided, 1)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            T res = identity_val;
            for(ENT j = segment_data->row_ptr[row]; j < segment_data->row_ptr[row + 1]; j++)
            {
                VNT col = segment_data->col_ids[j];
                T val = segment_data->vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = add_op(y_vals[row], res);
        }
    }
    t2 = omp_get_wtime();
    cout << "sparse BW: " << _matrix->sparse_segment.nnz * (2*sizeof(T) + sizeof(VNT))/((t2 - t1)*1e9) << " GB/s" << endl;


    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

