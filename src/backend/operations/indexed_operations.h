#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"


namespace lablas {
namespace backend {


template <typename M, typename LambdaOp>
LA_Info indexed_dense_vector_op(const Vector<M>* _mask,
    const Index* _indexes,
    const Index _nindexes,
    const Index _vector_size,
    LambdaOp&& _lambda_op,
    Descriptor* _desc)
{
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
                _lambda_op(idx);
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            _lambda_op(idx);
        }
    }
    return GrB_SUCCESS;
}

template <typename M, typename I, typename LambdaOp>
LA_Info indexed_dense_vector_op(const Vector<M>* _mask,
                                const Vector<I>* _indexes,
                                const Index _nindexes,
                                const Index _vector_size,
                                LambdaOp&& _lambda_op,
                                Descriptor* _desc)
                                {
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
                _lambda_op(idx);
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _nindexes; i++)
        {
            const Index idx = static_cast<Index>(_indexes->getDense()->get_vals()[i]);
            _lambda_op(idx);
        }
    }
    return GrB_SUCCESS;
                                }

}
}