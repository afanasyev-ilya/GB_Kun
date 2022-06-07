#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

namespace lablas {
namespace backend {

template <typename M, typename LambdaOp>
LA_Info generic_dense_vector_op_assign(const Vector<M>* _mask,
                                       const Index _size,
                                       LambdaOp &&_lambda_op,
                                       Descriptor *_desc)
{
    LOG_TRACE("Running generic_dense_vector_op_assign")
    if (_mask != NULL)
    {
        if (_mask->get_size() != _size)
            return GrB_DIMENSION_MISMATCH;

        Desc_value val;
        _desc->get(GrB_MASK, &val);
        if (_mask->is_sparse())
        {
            if(val == GrB_DEFAULT || val == GrB_STRUCTURE)
            {
                const Index* mask_ids = _mask->getSparse()->get_ids();
                const Index mask_nvals = _mask->getSparse()->get_nvals();
                #pragma omp parallel for
                for (Index i = 0; i < mask_nvals; i++)
                {
                    Index idx = mask_ids[i];
                    _lambda_op(idx, i);
                }
            }
            else if (val == GrB_STR_COMP)
            {
                const M* mask_data = _mask->getDense()->get_vals();
                #pragma omp parallel for
                for (Index i = 0; i < _size; i++)
                {
                    if (!mask_data[i])
                        _lambda_op(i, i);
                }
            }
        }
        else
        {
            const M* mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for (Index i = 0; i < _size; i++)
            {
                if (!mask_data[i] && val == GrB_STR_COMP || mask_data[i] && (val == GrB_DEFAULT || val == GrB_STRUCTURE))
                    _lambda_op(i, i);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _size; i++)
        {
            _lambda_op(i, i);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename M, typename LambdaOp>
LA_Info generic_dense_vector_op_extract(const Vector<M>* _mask,
                                        const Index _size,
                                        LambdaOp &&_lambda_op,
                                        Descriptor *_desc)
{
    LOG_TRACE("Running generic_dense_vector_op_extract")
    if (_mask != NULL)
    {
        if (_mask->get_size() != _size)
            return GrB_DIMENSION_MISMATCH;

        if (_mask->is_sparse())
        {
            const Index* mask_ids = _mask->getSparse()->get_ids();
            const Index mask_nvals = _mask->getSparse()->get_nvals();
            #pragma omp parallel for
            for (Index i = 0; i < mask_nvals; i++)
            {
                Index idx = mask_ids[i];
                _lambda_op(i, idx);
            }
        }
        else
        {
            const M* mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for (Index i = 0; i < _size; i++)
            {
                if (mask_data[i])
                    _lambda_op(i, i);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _size; i++)
        {
            _lambda_op(i, i);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename M, typename LambdaOp>
LA_Info generic_dense_vector_op(const Vector<M>* _mask,
    const Index _size,
    LambdaOp&& _lambda_op,
    Descriptor* _desc)
{
    LOG_TRACE("Running generic_dense_vector_op")
    if (_mask != NULL)
    {
        if (_mask->get_size() != _size)
            return GrB_DIMENSION_MISMATCH;

        if (_mask->is_sparse())
        {
            const Index* mask_ids = _mask->getSparse()->get_ids();
            const Index mask_nvals = _mask->getSparse()->get_nvals();
            #pragma omp parallel for
            for (Index i = 0; i < mask_nvals; i++)
            {
                Index idx = mask_ids[i];
                _lambda_op(idx);
            }
        }
        else
        {
            const M* mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for (Index i = 0; i < _size; i++)
            {
                if (mask_data[i])
                    _lambda_op(i);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index i = 0; i < _size; i++)
        {
            _lambda_op(i);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename LambdaOp, typename MonoidOpT>
LA_Info generic_dense_reduce_op(T* _tmp_val,
                                const Index _size,
                                LambdaOp &&_lambda_op,
                                MonoidOpT _monoid_op,
                                Descriptor *_desc)
{
    LOG_TRACE("Running generic_dense_reduce_op")
    #pragma omp parallel
    {
        T local_res = _monoid_op.identity();
        #pragma omp for
        for (Index i = 0; i < _size; i++)
        {
            local_res = _monoid_op(_lambda_op(i), local_res);
        }

        #pragma omp critical
        {
            *_tmp_val = _monoid_op(*_tmp_val, local_res);
        };
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename V, typename MonoidOpT>
LA_Info generic_sparse_vals_reduce_op(T *_tmp_val,
                                      const V*_vals,
                                      const Index _nvals,
                                      MonoidOpT _monoid_op,
                                      Descriptor *_desc)
{
    LOG_TRACE("Running generic_sparse_vals_reduce_op")
    #pragma omp parallel
    {
        T local_res = _monoid_op.identity();
        #pragma omp for
        for (Index i = 0; i < _nvals; i++)
        {
            T val = _vals[i];
            local_res = _monoid_op(val, local_res);
        }

        #pragma omp critical
        {
            *_tmp_val = _monoid_op(*_tmp_val, local_res);
        };
    }
    return GrB_SUCCESS;
}

}
}