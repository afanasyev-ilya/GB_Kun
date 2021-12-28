#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T, typename SemiringT>
void SpMV(MatrixSortCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y,
          SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel
    {
        #pragma omp for schedule(static, 256)
        for(VNT row = 0; row < _matrix->size; row++)
        {
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                T val = _matrix->vals[j];
                y_vals[row] = add_op(y_vals[row], mul_op(val, x_vals[col])) ;
            }
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}