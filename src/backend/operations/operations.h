/// @file operations.h
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Backend basic operations
/// @details Implements basic GraphBLAS operations
/// @date June 8, 2022

#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "generic_operations.h"
#include "indexed_operations.h"
#include "../../cpp_graphblas/types.hpp"


namespace lablas{

/// @namespace Lablas

namespace backend{

/// @namespace Backend

template <typename W, typename M, typename U, typename I, typename BinaryOpTAccum>
LA_Info assign(Vector<W>* _w,
               const Vector<M>* _mask,
               BinaryOpTAccum _accum,
               U _value,
               const Vector<I>* _indices,
               const Index _nindices,
               Descriptor* _desc)
{
    LOG_TRACE("Running assign with value-like vector variant")
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
    LOG_TRACE("Running assign with vector-like vector variant")
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
    LOG_TRACE("Running assign with value-like array variant")
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
    LOG_TRACE("Running assign with vector-like array variant")
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
    Desc_value algo;
    _desc->get(GrB_MXVMODE, &algo);
    if (algo == SPMV_GENERAL or (algo == GrB_DEFAULT and _u->is_dense()))
    {
        LOG_TRACE("Using SpMV General");
        backend::SpMV(_matrix, _u->getDense(), _w->getDense(), _desc, _accum, _op, _mask);
    }
    else
    {
        if (algo == SPMSPV_FOR or (algo == GrB_DEFAULT and _u->is_sparse())) {
            LOG_TRACE("Using SpMSpV for-based");
            backend::SpMSpV(_matrix, false, _u->getSparse(), _w->getDense(), _desc, _accum, _op, _mask);
        }
        if (algo == SPMSPV_MAP_TBB) {
            #ifdef __USE_TBB__
            LOG_TRACE("Using SpMSpV TBB map-based");
            SpMSpV_map_par(_matrix->get_csr(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
            #else
            LOG_TRACE("Using SpMSpV sequential map-based");
            SpMSpV_map_seq(_matrix->get_csr(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
            #endif
        }
        if (algo == SPMSPV_MAP_SEQ) {
            LOG_TRACE("Using SpMSpV sequential map-based");
            SpMSpV_map_seq(_matrix->get_csr(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
        }
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
    Desc_value algo;
    _desc->get(GrB_MXVMODE, &algo);
    if (algo == SPMV_GENERAL or (algo == GrB_DEFAULT and _u->is_dense()))
    {
        LOG_TRACE("Using SpMV General");
        double t1 = omp_get_wtime();
        backend::VSpM(_matrix, _u->getDense(), _w->getDense(), _desc, _accum, _op, _mask);
        double t2 = omp_get_wtime();
        SPMV_TIME += t2 - t1;
    }
    else
    {
        if (algo == SPMSPV_FOR or (algo == GrB_DEFAULT and _u->is_sparse()))
        {
            LOG_TRACE("Using SpMSpV for-based");
            double t1 = omp_get_wtime();
            backend::SpMSpV(_matrix, true, _u->getSparse(), _w->getDense(), _desc, _accum, _op, _mask);
            double t2 = omp_get_wtime();
            SPMSPV_TIME += t2 - t1;
        }
        if (algo == SPMSPV_BUCKET)
        {
            #ifdef __USE_TBB__
            LOG_TRACE("Using SpMSpV TBB map-based");
            SpMSpV_map_par(_matrix->get_csc(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
            #else
            LOG_TRACE("Using SpMSpV sequential map-based");
            SpMSpV_map_seq(_matrix->get_csc(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
            #endif
        }
        if (algo == SPMSPV_MAP_SEQ)
        {
            LOG_TRACE("Using SpMSpV sequential map-based");
            SpMSpV_map_seq(_matrix->get_csc(), _u->getSparse(), _w->getSparse(), _desc, _accum, _op, _mask);
        }
    }

    double t1 = omp_get_wtime();
    _w->convert_if_required();
    double t2 = omp_get_wtime();
    CONVERT_TIME += t2 - t1;

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
    LOG_TRACE("Running eWiseAdd")
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
    LOG_TRACE("Running eWiseMult")
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
    LOG_TRACE("Running value-like binary apply")
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
    LOG_TRACE("Running unary apply")
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
    LOG_TRACE("Running vector-like binary apply")
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

/// @brief Reduce Operation for Vector
///
/// Implements a reduce operation for vector which is basically does
/// w = op(w, u[i]) for each i. Accumulator also could be used as well as descriptor.
/// Different base operations are used for dense and sparse vectors.
/// @param[out] _val Pointer to result value
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Monoid operation
/// @param[in] _u Pointer to the Vector object
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info reduce(T *_val,
               BinaryOpTAccum _accum,
               MonoidT _op,
               const Vector<U> *_u,
               Descriptor *_desc)
{
    LOG_TRACE("Running vector-like reduce")
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

/// @brief Reduce Operation for Matrix
///
/// Implements a reduce operation for Matrix which is basically does
/// w = op(w, u[i, j]) for each i, j. Accumulator also could be used as well as descriptor.
/// @param[out] _val Pointer to result value
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Monoid operation
/// @param[in] _u Pointer to the Matrix object
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info reduce(T *_val,
               BinaryOpTAccum _accum,
               MonoidT _op,
               const Matrix<U> *_u,
               Descriptor *_desc)
{
    LOG_TRACE("Running matrix-like reduce")
    T reduce_result = _op.identity();
    Index nvals = _u->get_csr()->get_nnz();
    const U* u_vals = _u->get_csr()->get_vals();

    backend::generic_sparse_vals_reduce_op(&reduce_result, u_vals, nvals, _op, _desc);
    *_val = _accum(*_val, reduce_result);

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Extract Operation for Vector
///
/// Gather values in vector u from indices (vector index) and store in another
/// vector w. Assumes that we have allocated memory in w of size sizeof(indices)
/// w[i] = u[index[i]]
/// @param[out] w Pointer to result vector object
/// @param[in] mask Input mask
/// @param[in] accum Binary operation accumulator
/// @param[in] u Pointer to the input Vector object
/// @param[in] indices Pointer to the Indices Vector object
/// @param[in] desc Pointer to the descriptor
/// @result LA_Info status
template <typename W, typename M, typename U, typename I, typename BinaryOpT>
LA_Info extract(Vector<W>*       w,
                const Vector<M>* mask,
                BinaryOpT        accum,
                const Vector<U>* u,
                const Vector<I>* indices,
                Descriptor*      desc)
{
    LOG_TRACE("Running vector-like extract")
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

/// @brief MxM operation for two matrices
///
/// A backend implementation of MxM operation that selects the mxm algorithm from masked/unmasked type by mask
/// parameter and the specific implementation by passed descriptor. This function stores the result of multiplication
/// of two input matrices A and B in matrix by pointer C. Masked multiplication is done if the mask parameter is not
/// null pointer. Binary operation accumulator and semiring operations are supported as well with parameters accum and
/// op.
///
/// The algorithm chooses from following algorithms:
/// <li> Basic Masked IJK algorithm
/// <li> Masked IJK algorithm with both input matrices presorted
/// <li> Masked hash-based IKJ algorithm
/// <li> Unmasked hash-based IKJ algorithm
/// @param[out] C Pointer to the (empty) matrix object that will contain the result matrix.
/// @param[in] mask Pointer to the mask matrix
/// @param[in] accum NULL_TYPE accumulator
/// @param[in] op Semiring operation
/// @param[in] A Pointer to the first input matrix
/// @param[in] B Pointer to the second input matrix
/// @param[in] desc Pointer to the descriptor
/// @result LA_Info status
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
    Desc_value mask_mode;
    desc->get(GrB_MASK, &mask_mode);
    if (mask_mode == GrB_COMP || mask_mode == GrB_STR_COMP) {
        throw "Error: complementary mask is not supported yet";
    }
    Desc_value multiplication_mode;
    desc->get(GrB_MXMMODE, &multiplication_mode);
    if (mask) {
        if (multiplication_mode == GrB_IJK || multiplication_mode == GrB_IJK_DOUBLE_SORT) {
            bool a_is_sorted = (multiplication_mode == GrB_IJK_DOUBLE_SORT);
            if (a_is_sorted) {
                LOG_TRACE("Using double sort masked IJK method")
            } else {
                LOG_TRACE("Using single sort masked IJK method")
            }

            backend::SpMSpM_ijk(A,
                                B,
                                C,
                                mask,
                                op,
                                a_is_sorted);
        } else if (multiplication_mode == GrB_IKJ_MASKED) {
            LOG_TRACE("Using masked IKJ method")
            backend::SpMSpM_masked_ikj(mask,
                                       A,
                                       B,
                                       C,
                                       op);
        } else {
            return GrB_INVALID_VALUE;
        }
    } else {
        LOG_TRACE("Using unmasked hash based mxm method")
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
