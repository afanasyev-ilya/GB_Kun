#pragma once

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV_generic_cmp_logical_or_and(const Matrix<A> *_matrix,
                                       const DenseVector<X> *_x,
                                       DenseVector<Y> *_y,
                                       Descriptor *_desc,
                                       BinaryOpTAccum _accum,
                                       SemiringT _op,
                                       const Vector<M> *_mask)
{

}
