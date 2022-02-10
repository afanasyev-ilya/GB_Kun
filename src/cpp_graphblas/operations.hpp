#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool not_initialized(T const &_val)
{
    if(_val == NULL)
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename L, typename R, typename... Args>
bool not_initialized(L const& lhs, R const &rhs, Args const&... args)
{
    if(lhs == NULL || not_initialized(rhs,args...))
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename L, typename R, typename... Args>
bool dims_mismatched(L const& lhs, R const &rhs)
{
    if(lhs->get_vector()->get_size() != rhs->get_vector()->get_size())
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename L, typename R, typename... Args>
bool dims_mismatched(L const& lhs, R const &rhs, Args const&... args)
{
    if((lhs->get_vector()->get_size() != rhs->get_vector()->get_size()) || dims_mismatched(lhs,args...))
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseAdd(lablas::Vector<W>* _w,
                 const lablas::Vector<M>* _mask,
                 BinaryOpTAccum _accum,
                 BinaryOpT _op,
                 const lablas::Vector<U>* _u,
                 const lablas::Vector<V>* _v,
                 lablas::Descriptor* _desc)
{
    if(not_initialized(_w, _u, _v))
        return GrB_UNINITIALIZED_OBJECT;

    if(dims_mismatched(_w, _u, _v))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::eWiseAdd(_w->get_vector(), mask_t, _accum, _op, _u->get_vector(), _v->get_vector(), desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]); w is INTERSECTION of u an v */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseMult(Vector<W>* _w,
                  const Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const Vector<U>* _u,
                  const Vector<V>* _v,
                  Descriptor* _desc)
{
    if(not_initialized(_w, _u, _v))
        return GrB_UNINITIALIZED_OBJECT;

    if(dims_mismatched(_w, _u, _v))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::eWiseMult(_w->get_vector(), mask_t, _accum, _op, _u->get_vector(), _v->get_vector(), desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W>* _w,
              const Vector<M>* _mask,
              BinaryOpTAccum _accum,
              BinaryOpT _op,
              const T _val,
              const Vector<U>* _u,
              Descriptor* _desc)
{
    if(not_initialized(_w, _u))
        return GrB_UNINITIALIZED_OBJECT;

    if(dims_mismatched(_w, _u))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::apply(_w->get_vector(), mask_t, _accum, _op, _val, _u->get_vector(), desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ unary_op(u[i]) */
template <typename W, typename M, typename U, typename BinaryOpTAccum, typename UnaryOpT>
LA_Info apply(Vector<W>* _w,
              const Vector<M>* _mask,
              BinaryOpTAccum _accum,
              UnaryOpT _op,
              const Vector<U>* _u,
              Descriptor* _desc)
{
    if(not_initialized(_w, _u))
        return GrB_UNINITIALIZED_OBJECT;

    if(dims_mismatched(_w, _u))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::apply(_w->get_vector(), mask_t, _accum, _op, _u->get_vector(), desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W>* _w,
              const Vector<M>* _mask,
              BinaryOpTAccum _accum,
              BinaryOpT _op,
              const Vector<U>* _u,
              const T _val,
              Descriptor* _desc)
{
    if(not_initialized(_w, _u))
        return GrB_UNINITIALIZED_OBJECT;

    if(dims_mismatched(_w, _u))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::apply(_w->get_vector(), mask_t, _accum, _op,  _u->get_vector(), _val, desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ value */
template <typename W, typename M, typename U,
        typename BinaryOpTAccum>
LA_Info assign(Vector<W>*       _w,
               const Vector<M>* _mask,
               BinaryOpTAccum _accum,
               U _value,
               const Index *_indices,
               const Index _nindices,
               Descriptor*  _desc)
{
    if(not_initialized(_w))
        return GrB_UNINITIALIZED_OBJECT;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    LA_Info info = backend::assign(_w->get_vector(), mask_t, _accum, _value, _indices, _nindices, desc_t);
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ _u[indexes[i]] */
template <typename W, typename M, typename U,
        typename BinaryOpTAccum>
LA_Info assign(Vector<W>*       _w,
               const Vector<M>* _mask,
               BinaryOpTAccum _accum,
               Vector<U>* _u,
               const Index *_indices,
               const Index _nindices,
               Descriptor*  _desc)
{
    if(not_initialized(_w))
        return GrB_UNINITIALIZED_OBJECT;
    if(dims_mismatched(_w, _u))
        return GrB_DIMENSION_MISMATCH;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    LA_Info info = backend::assign(_w->get_vector(), mask_t, _accum, _u->get_vector(), _indices, _nindices, desc_t);
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info mxv (Vector<W>*       _w,
             const Vector<M>* _mask,
             BinaryOpTAccum   _accum,
             SemiringT        _op,
             const Matrix<a>* _matrix,
             const Vector<U>* _u,
             Descriptor*      _desc)
{
    if(not_initialized(_w, _u, _matrix, _desc))
        return GrB_UNINITIALIZED_OBJECT;

    auto mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    return backend::mxv(_w->get_vector(), mask_t, _accum, _op, _matrix->get_matrix(), _u->get_vector(), _desc->get_descriptor());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info vxm (Vector<W>*       _w,
             const Vector<M>* _mask,
             BinaryOpTAccum        _accum,
             SemiringT        _op,
             const Vector<U>* _u,
             const Matrix<a>* _matrix,
             Descriptor*      _desc)
{
    if(not_initialized(_w, _u, _matrix, _desc))
        return GrB_UNINITIALIZED_OBJECT;

    auto mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    return backend::vxm<W, M, a, U, BinaryOpTAccum, SemiringT>(_w->get_vector(), mask_t, _accum, _op, _matrix->get_matrix(), _u->get_vector(), _desc->get_descriptor());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w = op(w, u[i]) for each i; */
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info reduce(T *_val,
               BinaryOpTAccum _accum,
               MonoidT _op,
               const Vector<U>* _u,
               Descriptor* _desc)
{
    if(not_initialized(_u))
        return GrB_UNINITIALIZED_OBJECT;

    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    return backend::reduce(_val, _accum, _op, _u->get_vector(), desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*!
 * Extension method
 * Gather values in vector u from indices (vector index) and store in another
 * vector w.
 *   w[i] = u[index[i]]
 */

template <typename W, typename M, typename U, typename I, typename BinaryOpT>
LA_Info extract(Vector<W>*       w,
               const Vector<M>* mask,
               BinaryOpT        accum,
               const Vector<U>* u,
               const Vector<I>* indices,
               Descriptor*      desc) {
    if (u == NULL || w == NULL || indices == NULL || desc == NULL)
        return GrB_UNINITIALIZED_OBJECT;

    auto mask_t = (mask == NULL) ? NULL : mask->get_vector();
    auto desc_t = (desc == NULL) ? NULL : desc->get_descriptor();

    return backend::extract(w->get_vector(), mask_t, accum, u->get_vector(),
                                  indices->get_vector(), desc_t);
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


