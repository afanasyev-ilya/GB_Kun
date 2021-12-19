#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T, typename SemiringT>
void SpMV(const MatrixCOO<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    const T * __restrict x_vals = _x->get_vals();
    T * __restrict y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ENT block_start = _matrix->thread_bottom_border[tid];
        ENT block_end = _matrix->thread_top_border[tid];
        #pragma simd
        #pragma ivdep
        #pragma vector
        for(ENT i = block_start; i < block_end; i++)
        {
            VNT row = _matrix->row_ids[i];
            VNT col = _matrix->col_ids[i];
            T val = _matrix->vals[i];
            y_vals[row] = add_op(y_vals[row], mul_op(val, x_vals[col])) ;
        }
    };
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
