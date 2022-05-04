#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const T _val,
                  const lablas::Vector<U>* _u,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _val, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename BinaryOpTAccum, typename UnaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  UnaryOpT _op,
                  const lablas::Vector<U>* _u,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], val) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const lablas::Vector<U>* _u,
                  const T _val,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _u, _val, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ value */
template <typename W, typename M, typename U,
        typename BinaryOpTAccum>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpTAccum        _accum,
                   U _value,
                   const GrB_Index *_indices,
                   const GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{
    return lablas::assign(_w, _mask, _accum, _value, _indices, _nindices, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ _u[indexes[i]] */
template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpTAccum        _accum,
                   lablas::Vector<U>* _u,
                   const GrB_Index *_indices,
                   const GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{
    return lablas::assign(_w, _mask, _accum, _u, _indices, _nindices, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]); w is UNION of u an v */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_eWiseAdd(lablas::Vector<W>* _w,
                     const lablas::Vector<M>* _mask,
                     BinaryOpTAccum _accum,
                     BinaryOpT _op,
                     const lablas::Vector<U>* _u,
                     const lablas::Vector<V>* _v,
                     lablas::Descriptor* _desc)
{
    return lablas::eWiseAdd(_w, _mask, _accum, _op, _u, _v, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]); w is INTERSECTION of u an v */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_eWiseMult(lablas::Vector<W>* _w,
                      const lablas::Vector<M>* _mask,
                      BinaryOpTAccum _accum,
                      BinaryOpT _op,
                      const lablas::Vector<U>* _u,
                      const lablas::Vector<V>* _v,
                      lablas::Descriptor* _desc)
{
    return lablas::eWiseMult(_w, _mask, _accum, _op, _u, _v, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_vxm (lablas::Vector<W>*       _w,
                 const lablas::Vector<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Vector<U>* _u,
                 const lablas::Matrix<a>* _matrix,
                 lablas::Descriptor*      _desc)
{
    return lablas::vxm(_w, _mask, _accum, _op, _u, _matrix, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_mxm (lablas::Matrix<W>*       _c,
                 const lablas::Matrix<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Matrix<U>* _a,
                 const lablas::Matrix<a>* _b,
                 lablas::Descriptor*      _desc)
{
    return lablas::mxm(_c, _mask, _accum, _op, _a, _b, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_mxv (lablas::Vector<W>*       _w,
                 const lablas::Vector<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Matrix<a>* _matrix,
                 const lablas::Vector<U>* _u,
                 lablas::Descriptor*      _desc)
{
    return lablas::mxv(_w, _mask, _accum, _op, _matrix, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w = op(w, u[i]) for each i; */
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info GrB_reduce(T *_val,
                   BinaryOpTAccum _accum,
                   MonoidT _op,
                   const lablas::Vector<U>* _u,
                   lablas::Descriptor* _desc)
{
    return lablas::reduce(_val, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info GrB_reduce(T *_val,
                   BinaryOpTAccum _accum,
                   MonoidT _op,
                   const lablas::Matrix<U>* _u,
                   lablas::Descriptor* _desc)
{
    return lablas::reduce(_val, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
LA_Info GrB_Vector_clear(lablas::Vector<T>* _vec)
{
    return _vec->clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
LA_Info GrB_select(lablas::Vector<W> *w,
                   const lablas::Vector<M> *mask,
                   BinaryOpT accum,
                   SelectOpT op,
                   const lablas::Vector<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, accum, op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename SelectOpT>
LA_Info GrB_select(lablas::Vector<W> *w,
                   const lablas::Vector<M> *mask,
                   NULL_TYPE accum,
                   SelectOpT op,
                   const lablas::Vector<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, lablas::second<W, T>(), op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename SelectOpT>
LA_Info GrB_select(lablas::Matrix<W> *w,
                   const lablas::Matrix<M> *mask,
                   NULL_TYPE accum,
                   SelectOpT op,
                   const lablas::Matrix<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, lablas::second<W, T>(), op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
