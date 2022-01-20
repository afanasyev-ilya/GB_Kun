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

    Index vector_size = _w->get_vector()->getDense()->get_size();
    auto w_vals = _w->get_vector()->getDense()->get_vals();
    auto u_vals = _u->get_vector()->getDense()->get_vals();
    auto v_vals = _v->get_vector()->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, v_vals, &_op] (Index idx)
    {
        w_vals[idx] = _op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
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

    Index vector_size = _w->get_vector()->getDense()->get_size();
    auto w_vals = _w->get_vector()->getDense()->get_vals();
    auto u_vals = _u->get_vector()->getDense()->get_vals();
    auto v_vals = _v->get_vector()->getDense()->get_vals();

    if(true) /* vectors are dense */
    {
        auto lambda_op = [w_vals, u_vals, v_vals, &_op] (Index idx)
        {
            w_vals[idx] = _op(u_vals[idx], v_vals[idx]);
        };

        return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }

    return GrB_SUCCESS;
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

    Index vector_size = _w->get_vector()->getDense()->get_size();
    auto w_vals = _w->get_vector()->getDense()->get_vals();
    auto u_vals = _u->get_vector()->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, _val, &_op] (Index idx)
    {
        w_vals[idx] = _op(_val, u_vals[idx]);
    };

    return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
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

    Index vector_size = _w->get_vector()->getDense()->get_size();
    auto w_vals = _w->get_vector()->getDense()->get_vals();
    auto u_vals = _u->get_vector()->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_op] (Index idx)
    {
        w_vals[idx] = _op(u_vals[idx]);
    };

    return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
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

    Index vector_size = _w->get_vector()->getDense()->get_size();
    auto w_vals = _w->get_vector()->getDense()->get_vals();
    auto u_vals = _u->get_vector()->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, _val, &_op] (Index idx)
    {
        w_vals[idx] = _op(u_vals[idx], _val);
    };

    return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
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

    _w->get_vector()->force_to_dense();
    W* w_vals = _w->get_vector()->getDense()->get_vals(); // can be called since force dense conversion before
    Index vector_size = _w->get_vector()->getDense()->get_size();

    auto lambda_op = [w_vals, _value] (Index idx)
    {
        w_vals[idx] = _value;
    };
    LA_Info info;
    if(_indices == NULL)
    {
        info = backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }
    else
    {
        info = backend::indexed_dense_vector_op(mask_t, _indices, _nindices, vector_size, lambda_op, desc_t);
    }
    _w->get_vector()->convert_if_required();
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

    _w->get_vector()->force_to_dense();
    Index vector_size = _w->get_vector()->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->get_vector()->getDense()->get_vals();
    U* u_vals = _u->get_vector()->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_accum](Index idx) {
        w_vals[idx] = _accum(w_vals[idx], u_vals[idx]);
    };
    LA_Info info;
    if(_indices == NULL)
    {
        info = backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }
    else
    {
        info = backend::indexed_dense_vector_op(mask_t, _indices, _nindices, vector_size, lambda_op, desc_t);
    }
    _w->get_vector()->convert_if_required();
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

    if(_u->get_vector()->is_dense())
        backend::SpMV(_matrix->get_matrix(), _u->get_vector()->getDense(), _w->get_vector()->getDense(), _desc->get_descriptor(), _accum, _op, mask_t);
    else
    {
        backend::SpMV(_matrix->get_matrix(), _u->get_vector()->getDense(), _w->get_vector()->getDense(), _desc->get_descriptor(), _accum, _op, mask_t);
    }

    return GrB_SUCCESS;
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
    backend::VSpM(_matrix->get_matrix(), _u->get_vector()->getDense(), _w->get_vector()->getDense(), _desc->get_descriptor(), _accum, _op, mask_t);

    return GrB_SUCCESS;
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

    Index vector_size = _u->get_vector()->getDense()->get_size();
    const U* u_vals = _u->get_vector()->getDense()->get_vals();

    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    auto lambda_op = [u_vals] (Index idx)->U
    {
        return u_vals[idx];
    };

    T reduce_result = _op.identity();

    backend::generic_dense_reduce_op(&reduce_result, vector_size, lambda_op, _op, desc_t);

    *_val = _accum(*_val, reduce_result);

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


