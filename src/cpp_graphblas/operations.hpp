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
    if(lhs->get_vector()->getDense()->get_size() != rhs->get_vector()->getDense()->get_size())
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename L, typename R, typename... Args>
bool dims_mismatched(L const& lhs, R const &rhs, Args const&... args)
{
    if((lhs->get_vector()->getDense()->get_size() != rhs->get_vector()->getDense()->get_size()) || dims_mismatched(lhs,args...))
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
        typename BinaryOpT>
LA_Info assign(Vector<W>*       _w,
               const Vector<M>* _mask,
               BinaryOpT        _accum,
               const U _value,
               const Index *_indices,
               const Index _nindices,
               Descriptor*  _desc)
{
    if(not_initialized(_w))
        return GrB_UNINITIALIZED_OBJECT;

    auto                 mask_t = (_mask == NULL) ? NULL : _mask->get_vector();
    backend::Descriptor* desc_t = (_desc == NULL) ? NULL : _desc->get_descriptor();

    if(_indices == NULL)
    {
        Index vector_size = _w->get_vector()->getDense()->get_size();
        auto w_vals = _w->get_vector()->getDense()->get_vals();

        auto lambda_op = [w_vals, _value] (Index idx)
        {
            w_vals[idx] = _value;
        };
        return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }
    else
    {
        throw "Error in assign :  _indices != NULL currently not supported";
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpT, typename SemiringT>
LA_Info mxv (Vector<W>*       w,
             const Vector<M>* mask,
             BinaryOpT        accum,
             SemiringT        op,
             const Matrix<a>* A,
             const Vector<U>* u,
             Descriptor*      desc)
{

    if (w == NULL || u == NULL || A == NULL || desc == NULL) {
        return GrB_UNINITIALIZED_OBJECT;
    }

    if(mask != NULL)
    {
        backend::SpMV(A->get_matrix(), u->get_vector(), w->get_vector(), desc->get_descriptor(), op,
                      mask->get_vector());
    }
    else
    {
        backend::SpMV<W, M, SemiringT>(A->get_matrix(), u->get_vector(), w->get_vector(), desc->get_descriptor(), op,
                                       NULL);
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpT, typename SemiringT>
LA_Info vxm (Vector<W>*       w,
             const Vector<M>* mask,
             BinaryOpT        accum,
             SemiringT        op,
             const Matrix<a>* A,
             const Vector<U>* u,
             Descriptor*      desc)
{
    if (w == NULL || u == NULL || A == NULL || desc == NULL) {
        return GrB_UNINITIALIZED_OBJECT;
    }

    if(mask != NULL)
        backend::VSpM(A->get_matrix(), u->get_vector(), w->get_vector(), desc->get_descriptor(),  op, mask->get_vector());
    else
        backend::VSpM<W, M, SemiringT>(A->get_matrix(), u->get_vector(), w->get_vector(), desc->get_descriptor(), op, NULL);
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


