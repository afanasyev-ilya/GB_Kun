#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "generic_operations.h"
#include "indexed_operations.h"


namespace lablas{
namespace backend{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    U _u,
    const Index* _indices,
    const Index _nindices,
    Descriptor* _desc) {

    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_accum](Index idx) {
        w_vals[idx] = _accum(w_vals[idx], _u);
    };

    LA_Info info;
    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    Vector<U>* _u,
    const Index* _indices,
    const Index _nindices,
    Descriptor* _desc) {

    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();
    U* u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_accum](Index idx) {
        w_vals[idx] = _accum(w_vals[idx], u_vals[idx]);
    };

    LA_Info info;
    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseAdd(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    BinaryOpT _op,
    const Vector<U>* _u,
    const Vector<V>* _v,
    Descriptor* _desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();
    auto v_vals = _v->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, v_vals, &_op](Index idx)
    {
        w_vals[idx] = _op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseMult(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    BinaryOpT _op,
    const Vector<U>* _u,
    const Vector<V>* _v,
    Descriptor* _desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();
    auto v_vals = _v->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, v_vals, &_op](Index idx)
    {
        w_vals[idx] = _op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    BinaryOpT _op,
    const T _val,
    const Vector<U>* _u,
    Descriptor* _desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, _val, &_op](Index idx)
    {
        w_vals[idx] = _op(_val, u_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename BinaryOpTAccum, typename UnaryOpT>
LA_Info apply(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    UnaryOpT _op,
    const Vector<U>* _u,
    Descriptor* _desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_op](Index idx)
    {
        w_vals[idx] = _op(u_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W>* _w,
    const Vector<M>* _mask,
    BinaryOpTAccum _accum,
    BinaryOpT _op,
    const Vector<U>* _u,
    const T _val,
    Descriptor* _desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, _val, &_op](Index idx)
    {
        w_vals[idx] = _op(u_vals[idx], _val);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w = op(w, u[i]) for each i; */
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info reduce(T* _val,
    BinaryOpTAccum _accum,
    MonoidT _op,
    const Vector<U>* _u,
    Descriptor* _desc) {


    Index vector_size = _u->getDense()->get_size();
    const U* u_vals = _u->->getDense()->get_vals();

    auto lambda_op = [u_vals](Index idx)->U
    {
        return u_vals[idx];
    };

    T reduce_result = _op.identity();

    backend::generic_dense_reduce_op(&reduce_result, vector_size, lambda_op, _op, desc_t);

    *_val = _accum(*_val, reduce_result);

    return GrB_SUCCESS;
}

}
}