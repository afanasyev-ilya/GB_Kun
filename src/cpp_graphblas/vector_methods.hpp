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
                 BinaryOpT _binary_op,
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

    auto lambda_op = [w_vals, u_vals, v_vals, &_binary_op] (Index idx)
    {
        w_vals[idx] = _binary_op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
