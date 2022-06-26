#pragma once

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV_map_cmp_logical_or_and(const MatrixCSR<A> *_matrix,
                const SparseVector <X> *_x,
                SparseVector <Y> *_y,
                Descriptor *_desc,
                BinaryOpTAccum _accum,
                SemiringT _op,
                const Vector <M> *_mask)
{

}
