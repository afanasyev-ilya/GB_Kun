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
        info = backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }
    else
    {
        info = backend::indexed_dense_vector_op(mask_t, _indices, _nindices, vector_size, lambda_op, desc_t);
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
        info = backend::generic_dense_vector_op(mask_t, vector_size, lambda_op, desc_t);
    }
    else
    {
        info = backend::indexed_dense_vector_op(mask_t, _indices, _nindices, vector_size, lambda_op, desc_t);
    }
    _w->convert_if_required();
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}