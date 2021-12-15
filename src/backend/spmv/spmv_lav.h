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
    for(VNT cur_seg = 0; cur_seg < dense_segments; cur_seg++)
    {
        VNT num_rows = _matrix->size;
        #pragma omp parallel for schedule(static)
        for(VNT row = 0; row < num_rows; row++)
        {
            T res = identity_val;
            for(ENT j = _matrix->dense_row_ptr[cur_seg][row]; j < _matrix->dense_row_ptr[cur_seg][row + 1]; j++)
            {
                VNT col = _matrix->dense_col_ids[cur_seg][j];
                T val = _matrix->dense_vals[cur_seg][j];
                res = add_op(res, mul_op(val, x_vals[col]));
            }
            y_vals[row] = res;
        }
    }

    VNT num_rows = _matrix->size;
    #pragma omp parallel for schedule(static)
    for(VNT row = 0; row < num_rows; row++)
    {
        T res = identity_val;
        for(ENT j = _matrix->sparse_row_ptr[row]; j < _matrix->sparse_row_ptr[row + 1]; j++)
        {
            VNT col = _matrix->sparse_col_ids[j];
            T val = _matrix->sparse_vals[j];
            res = add_op(res, mul_op(val, x_vals[col]));
        }
        y_vals[row] = res;
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

