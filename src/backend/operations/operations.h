#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "generic_operations.h"
#include "indexed_operations.h"
#include "../../cpp_graphblas/types.hpp"


namespace lablas{
namespace backend{

template <typename W, typename M, typename U, typename I, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
               const Vector<M>* _mask,
               BinaryOpTAccum _accum,
               U _value,
               const Vector<I>* _indices,
               const Index _nindices,
               Descriptor* _desc)
{
    LA_Info info;
    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();

    auto lambda_op = [w_vals, _value] (Index idx1, Index idx2) {
        w_vals[idx1] = _value;
    };

    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op_assign(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op_assign(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename I, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
               const Vector<M>* _mask,
               BinaryOpTAccum _accum,
               Vector<U>* _u,
               const Vector<I>* _indices,
               const Index _nindices,
               Descriptor* _desc)
{
    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();
    U* u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_accum](Index idx1, Index idx2) {
        w_vals[idx1] = _accum(w_vals[idx1], u_vals[idx2]);
    };

    LA_Info info;
    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op_assign(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op_assign(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
               const Vector<M> *_mask,
               BinaryOpTAccum _accum,
               U _value,
               const Index *_indices,
               const Index _nindices,
               Descriptor *_desc)
{
    LA_Info info = GrB_SUCCESS;
    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();

    auto lambda_op = [w_vals, _value] (Index idx1, Index idx2) {
        w_vals[idx1] = _value;
    };

    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op_assign(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op_assign(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info assign(Vector<W> *_w,
               const Vector<M> *_mask,
               BinaryOpTAccum _accum,
               Vector<U> *_u,
               const Index *_indices,
               const Index _nindices,
               Descriptor *_desc)
{
    _w->force_to_dense();

    Index vector_size = _w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = _w->getDense()->get_vals();
    U* u_vals = _u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &_accum](Index idx1, Index idx2) {
        w_vals[idx1] = _accum(w_vals[idx1], u_vals[idx2]);
    };

    LA_Info info;
    if (_indices == NULL)
    {
        info = backend::generic_dense_vector_op_assign(_mask, vector_size, lambda_op, _desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op_assign(_mask, _indices, _nindices, vector_size, lambda_op, _desc);
    }
    _w->convert_if_required();
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// w += Au
template <typename W, typename M, typename A, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info mxv (Vector<W>*       _w,
             const Vector<M>* _mask,
             BinaryOpTAccum   _accum,
             SemiringT        _op,
             const Matrix<A>* _matrix,
             const Vector<U>* _u,
             Descriptor*      _desc)
{
    bool switch_cond = _u->is_dense();
    #ifdef __DISABLE_SPMSPV__
    switch_cond = true;
    #endif
    if(switch_cond)
    {
        GLOBAL_PERF_STATS(backend::SpMV(_matrix, _u->getDense(), _w->getDense(), _desc,
                                        _accum, _op, _mask), GLOBAL_SPMV_TIME);

    }
    else
    {
        GLOBAL_PERF_STATS(backend::SpMSpV(_matrix, false, _u->getSparse(), _w->getDense(),
                                          _desc, _accum, _op, _mask), GLOBAL_SPMSPV_TIME);
    }
    _w->convert_if_required();

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename A, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info vxm (Vector<W>*       _w,
             const Vector<M>* _mask,
             BinaryOpTAccum   _accum,
             SemiringT        _op,
             const Matrix<A>* _matrix,
             const Vector<U>* _u,
             Descriptor*      _desc)
{
    bool switch_cond = _u->is_dense();
    #ifdef __DISABLE_SPMSPV__
    switch_cond = true;
    #endif
    if(switch_cond)
    {
        GLOBAL_PERF_STATS(backend::VSpM(_matrix, _u->getDense(), _w->getDense(), _desc,
                                        _accum, _op, _mask), GLOBAL_SPMV_TIME);
    }
    else
    {
        if(_u->get_nvals() > 1000)
        {
            GLOBAL_PERF_STATS(backend::SpMSpV(_matrix, true, _u->getSparse(), _w->getDense(),
                                              _desc, _accum, _op, _mask), GLOBAL_SPMSPV_TIME);
        }
        else
        {
            GLOBAL_PERF_STATS(backend::SpMSpV_map_seq(_matrix->get_csr(), _u->getSparse(), _w->getSparse(),
                                                      _desc, _accum, _op, _mask), GLOBAL_SPMSPV_TIME);
        }
    }
    _w->convert_if_required();

    return GrB_SUCCESS;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseAdd(Vector<W> *_w,
                 const Vector<M> *_mask,
                 BinaryOpTAccum _accum,
                 BinaryOpT _op,
                 const Vector<U> *_u,
                 const Vector<V> *_v,
                 Descriptor *_desc)
{
    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();
    auto v_vals = _v->getDense()->get_vals();

    auto add_op = generic_extract_add(_op);

    auto lambda_op = [w_vals, u_vals, v_vals, &add_op](Index idx)
    {
        w_vals[idx] = add_op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info eWiseMult(Vector<W> *_w,
                  const Vector<M> *_mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const Vector<U> *_u,
                  const Vector<V> *_v,
                  Descriptor *_desc)
{

    Index vector_size = _w->getDense()->get_size();
    auto w_vals = _w->getDense()->get_vals();
    auto u_vals = _u->getDense()->get_vals();
    auto v_vals = _v->getDense()->get_vals();

    auto mull_op = generic_extract_mull(_op);

    auto lambda_op = [w_vals, u_vals, v_vals, &mull_op](Index idx)
    {
        w_vals[idx] = mull_op(u_vals[idx], v_vals[idx]);
    };

    return backend::generic_dense_vector_op(_mask, vector_size, lambda_op, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W> *_w,
              const Vector<M> *_mask,
              BinaryOpTAccum _accum,
              BinaryOpT _op,
              const T _val,
              const Vector<U> *_u,
              Descriptor *_desc)
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
LA_Info apply(Vector<W> *_w,
              const Vector<M> *_mask,
              BinaryOpTAccum _accum,
              UnaryOpT _op,
              const Vector<U> *_u,
              Descriptor *_desc)
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
LA_Info reduce(T *_val,
               BinaryOpTAccum _accum,
               MonoidT _op,
               const Vector<U> *_u,
               Descriptor *_desc)
{
    T reduce_result = _op.identity();
    if(_u->is_dense())
    {
        Index vector_size = _u->getDense()->get_size();
        const U* u_vals = _u->getDense()->get_vals();

        auto lambda_op = [u_vals](Index idx)->U
        {
            return u_vals[idx];
        };

        backend::generic_dense_reduce_op(&reduce_result, vector_size, lambda_op, _op, _desc);
    }
    else // is sparse
    {
        Index nvals = _u->getSparse()->get_nvals();
        const U* u_vals = _u->getSparse()->get_vals();

        backend::generic_sparse_vals_reduce_op(&reduce_result, u_vals, nvals, _op, _desc);
    }
    *_val = _accum(*_val, reduce_result);

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w = op(w, u[i]) for each i; */
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info reduce(T *_val,
               BinaryOpTAccum _accum,
               MonoidT _op,
               const Matrix<U> *_u,
               Descriptor *_desc)
{
    T reduce_result = _op.identity();
    Index nvals = _u->get_csr()->get_nnz();
    const U* u_vals = _u->get_csr()->get_vals();

    backend::generic_sparse_vals_reduce_op(&reduce_result, u_vals, nvals, _op, _desc);
    *_val = _accum(*_val, reduce_result);

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Assume that we have allocated memory in w of size sizeof(indices) */
template <typename W, typename M, typename U, typename I, typename BinaryOpT>
LA_Info extract(Vector<W>*       w,
                const Vector<M>* mask,
                BinaryOpT        accum,
                const Vector<U>* u,
                const Vector<I>* indices,
                Descriptor*      desc)
{
    w->force_to_dense();

    Index vector_size = w->getDense()->get_size(); // can be called since force dense conversion before
    W* w_vals = w->getDense()->get_vals();
    const U* u_vals = u->getDense()->get_vals();

    auto lambda_op = [w_vals, u_vals, &accum](Index idx1, Index idx2) {
        w_vals[idx1] = accum(w_vals[idx1], u_vals[idx2]);
    };

    LA_Info info;
    if (indices == NULL)
    {
        info = backend::generic_dense_vector_op_extract(mask, vector_size, lambda_op, desc);
    }
    else
    {
        info = backend::indexed_dense_vector_op_extract(mask, indices, indices->get_size(), vector_size, lambda_op, desc);
    }
    w->convert_if_required();
    return info;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename c, typename a, typename b, typename m,
        typename BinaryOpT,     typename SemiringT>
LA_Info mxm(Matrix<c>* C,
            const Matrix<m> *mask,
            BinaryOpT accum,
            SemiringT op,
            const Matrix<a> *A,
            const Matrix<b> *B,
            Descriptor *desc)
{
    Desc_value multiplication_mode;
    desc->get(GrB_MXMMODE, &multiplication_mode);
    if (mask) {
        bool a_is_sorted = (multiplication_mode == GrB_IJK_DOUBLE_SORT);
        backend::SpMSpM_ijk(A,
                            B,
                            C,
                            mask,
                            op,
                            a_is_sorted);
        /*
        if (multiplication_mode == GrB_IJK) {
            backend::SpMSpM_ijk(A,
                                B,
                                C,
                                mask,
                                op);
        } else if (multiplication_mode == GrB_IKJ) {
            backend::SpMSpM_masked_ikj(mask,
                                       A,
                                       B,
                                       C,
                                       op);
        } else {
            return GrB_INVALID_VALUE;
        }
         */
    } else {
        backend::SpMSpM_unmasked_ikj(A,
                                     B,
                                     C,
                                     op);
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
