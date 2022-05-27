#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"


namespace lablas {
namespace backend {

template <typename M, typename LambdaOp>
LA_Info indexed_dense_vector_op_assign(const Vector<M>* _mask,
    const Index* _indexes,
    const Index _nindexes,
    const Index _vector_size,
    LambdaOp&& _lambda_op,
    Descriptor* _desc)
{
    LOG_TRACE("Running indexed_dense_vector_op_assign (array variant)")
    if (_mask != NULL)
    {
        // TODO if mask is sparse

        if (_mask->get_size() != _vector_size)
            return GrB_DIMENSION_MISMATCH;

        const M* mask_data = _mask->getDense()->get_vals();

        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            if (mask_data[idx])
                _lambda_op(idx, i);
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            _lambda_op(idx, i);
        }
    }
    return GrB_SUCCESS;
}

template <typename M, typename I, typename LambdaOp>
LA_Info indexed_dense_vector_op_assign(const Vector<M>* _mask,
    const Vector<I>* _indexes,
    const Index _nindexes,
    const Index _vector_size,
    LambdaOp&& _lambda_op,
    Descriptor* _desc)
{
    LOG_TRACE("Running indexed_dense_vector_op_assign (vector variant)")
    auto ids = _indexes->getDense()->get_vals();
    if (_mask != NULL)
    {
        // TODO if mask is sparse

        if (_mask->get_size() != _vector_size)
            return GrB_DIMENSION_MISMATCH;

        const M* mask_data = _mask->getDense()->get_vals();

        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = static_cast<Index>(_indexes->getDense()->get_vals()[i]);
            if (mask_data[idx])
                _lambda_op(idx, i);
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = static_cast<Index>(ids[i]);
            _lambda_op(idx, i);
        }
    }
    return GrB_SUCCESS;
}

template <typename M, typename LambdaOp>
LA_Info indexed_dense_vector_op_extract(const Vector<M>* _mask,
                                const Index* _indexes,
                                const Index _nindexes,
                                const Index _vector_size,
                                LambdaOp&& _lambda_op,
                                Descriptor* _desc)
{
    LOG_TRACE("Running indexed_dense_vector_op_extract (array variant)")
    if (_mask != NULL)
    {
        // TODO if mask is sparse

        if (_mask->get_size() != _vector_size)
            return GrB_DIMENSION_MISMATCH;

        const M* mask_data = _mask->getDense()->get_vals();

#pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            if (mask_data[idx])
                _lambda_op(i, idx);
        }
    }
    else
    {
#pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            _lambda_op(i, idx);
        }
    }
    return GrB_SUCCESS;
}

template <typename M, typename I, typename LambdaOp>
LA_Info indexed_dense_vector_op_extract(const Vector<M>* _mask,
                                const Vector<I>* _indexes,
                                const Index _nindexes,
                                const Index _vector_size,
                                LambdaOp&& _lambda_op,
                                Descriptor* _desc)
{
    LOG_TRACE("Running indexed_dense_vector_op_extract (vector variant)")
    auto ids = _indexes->getDense()->get_vals();
    if (_mask != NULL)
    {
        // TODO if mask is sparse

        if (_mask->get_size() != _vector_size)
            return GrB_DIMENSION_MISMATCH;

        const M* mask_data = _mask->getDense()->get_vals();

#pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = static_cast<Index>(_indexes->getDense()->get_vals()[i]);
            if (mask_data[idx])
                _lambda_op(i, idx);
        }
    }
    else
    {
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = static_cast<Index>(ids[i]);
            _lambda_op(i, idx);
        }
    }
    return GrB_SUCCESS;
}

}
}