#pragma once

/*LA_Info GrB_apply (d, NULL, NULL, GrB_DIV_FP32, d_out, damping, NULL)
{
    return apply(Vector<W>*       w,
                 const Vector<M>* mask,
                 BinaryOpT        accum,
                 UnaryOpT         op,
                 const Vector<U>* u,
                 Descriptor*      desc);
}*/

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  const BinaryOpTAccum _accum,
                  const BinaryOpT _op,
                  const T val,
                  const lablas::Vector<U>* _u,
                  const lablas::Descriptor* _desc)
{
    return GrB_SUCCESS;
}

/* w[i] = mask[i] ^ op(u[i], val) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  const BinaryOpTAccum _accum,
                  const BinaryOpT _op,
                  const lablas::Vector<U>* _u,
                  const T val,
                  const lablas::Descriptor* _desc)
{
    return GrB_SUCCESS;
}

/* w[indexes[i]] = mask[indexes[i]] ^ value */
template <typename W, typename M, typename U,
        typename BinaryOpT>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpT        _accum,
                   const U _value,
                   const GrB_Index *_indices,
                   const GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{

    return lablas::assign(_w, _mask, _accum, _value, _indices, _nindices, _desc);
}

