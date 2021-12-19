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

    VNT dense_segments = _matrix->dense_segments;
    VNT num_rows = _matrix->size;
    #pragma omp parallel
    {
        int cur_seg = 0;
        const VNT *row_ids = _matrix->dense_vertex_groups[cur_seg].ptr();
        const VNT nnz_num_rows = _matrix->dense_vertex_groups[cur_seg].size();

        #pragma omp for schedule(guided, 1)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            T res = identity_val;
            for(ENT j = _matrix->dense_row_ptr[cur_seg][row]; j < _matrix->dense_row_ptr[cur_seg][row + 1]; j++)
            {
                VNT col = _matrix->dense_col_ids[cur_seg][j];
                T val = _matrix->dense_vals[cur_seg][j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = add_op(y_vals[row], res);
        }
    }

    #pragma omp parallel
    {
        for(VNT cur_seg = 1; cur_seg < dense_segments; cur_seg++)
        {
            const VNT *row_ids = _matrix->dense_vertex_groups[cur_seg].ptr();
            const VNT nnz_num_rows = _matrix->dense_vertex_groups[cur_seg].size();

            #pragma omp for schedule(guided, 1)
            for(VNT idx = 0; idx < nnz_num_rows; idx++)
            {
                VNT row = row_ids[idx];
                T res = identity_val;
                for(ENT j = _matrix->dense_row_ptr[cur_seg][row]; j < _matrix->dense_row_ptr[cur_seg][row + 1]; j++)
                {
                    VNT col = _matrix->dense_col_ids[cur_seg][j];
                    T val = _matrix->dense_vals[cur_seg][j];
                    res = add_op(res, mul_op(val, x_vals[col]));
                }
                y_vals[row] = add_op(y_vals[row], res);
            }
        }
    }

    #pragma omp parallel
    {
        const VNT *row_ids = _matrix->sparse_vertex_group.ptr();
        const VNT nnz_num_rows = _matrix->sparse_vertex_group.size();

        #pragma omp for schedule(guided, 1)
        for(VNT idx = 0; idx < nnz_num_rows; idx++)
        {
            VNT row = row_ids[idx];
            T res = identity_val;
            for(ENT j = _matrix->sparse_row_ptr[row]; j < _matrix->sparse_row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->sparse_col_ids[j];
                T val = _matrix->sparse_vals[j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = add_op(y_vals[row], res);
        }
    }

    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
    //reorder(y_vals, _matrix->new_to_old, _matrix_size);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

