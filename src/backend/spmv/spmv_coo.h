#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
void SpMV(const MatrixCOO<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          BinaryOpTAccum _accum,
          SemiringT _op,
          Workspace *_workspace)
{
    LOG_TRACE("Running SpMV for COO")
    const X * __restrict x_vals = _x->get_vals();
    Y * __restrict y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    Y *buffer = (Y*)_workspace->get_first_socket_vector();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ENT block_start = _matrix->thread_bottom_border[tid];
        ENT block_end = _matrix->thread_top_border[tid];

        #pragma omp for
        for(VNT row = 0; row < _matrix->size; row++)
        {
            buffer[row] = identity_val;
        }

        #pragma omp barrier

        #pragma simd
        #pragma ivdep
        #pragma vector
        for(ENT i = block_start; i < block_end; i++)
        {
            VNT row = _matrix->row_ids[i];
            VNT col = _matrix->col_ids[i];
            A val = _matrix->vals[i];
            buffer[row] = add_op(buffer[row], mul_op(val, x_vals[col])) ;
        }

        #pragma omp barrier

        #pragma omp for schedule(static)
        for(VNT row = 0; row < _matrix->size; row++)
        {
            y_vals[row] = _accum(y_vals[row], buffer[row]);
        }
    };
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
