#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV(MatrixSortCSR<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          BinaryOpTAccum _accum,
          SemiringT _op)
{
    LOG_TRACE("Running SpMV for SortCSR")
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    #pragma omp parallel
    {
        #pragma omp for schedule(static, 256)
        for(VNT row = 0; row < _matrix->size; row++)
        {
            Y res = identity_val;
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                A val = _matrix->vals[j];
                res = add_op(res, mul_op(val, x_vals[col])) ;
            }
            y_vals[row] = _accum(y_vals[row], res);
        }
    };

    //reorder(y_vals, _matrix->col_backward_conversion, _matrix->size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
