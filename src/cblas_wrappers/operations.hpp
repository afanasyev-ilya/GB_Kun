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

template <typename W, typename M, typename U,
        typename BinaryOpT>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpT        _accum,
                   const U _value,
                   const GrB_Index *_indices,
                   GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{
    return lablas::assign(_w, _mask, _accum, _value, _indices, _nindices, _desc);
}

