#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

namespace lablas{
namespace backend{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename M, typename LambdaOp>
LA_Info generic_dense_vector_op(const Vector<M> *_mask,
                                const Index _size,
                                LambdaOp &&_lambda_op,
                                Descriptor *_desc)
{
    if(_mask != NULL)
    {
        if(_mask->get_size() != _size)
            return GrB_DIMENSION_MISMATCH;

        if(_mask->is_sparse())
        {
            const Index *mask_ids = _mask->getSparse()->get_ids();
            const Index mask_nvals = _mask->getSparse()->get_nvals();
            #pragma omp parallel for
            for(Index i = 0; i < mask_nvals; i++)
            {
                Index idx = mask_ids[i];
                _lambda_op(idx);
            }
        }
        else
        {
            const M *mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for(Index i = 0; i < _size; i++)
            {
                if(mask_data[i])
                    _lambda_op(i);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for(Index i = 0; i < _size; i++)
        {
            _lambda_op(i);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename M, typename LambdaOp>
LA_Info indexed_dense_vector_op(const Vector<M> *_mask,
                                const Index *_indexes,
                                const Index _nindexes,
                                const Index _vector_size,
                                LambdaOp &&_lambda_op,
                                Descriptor *_desc)
{
    if(_mask != NULL)
    {
        // TODO if mask is sparse

        if(_mask->get_size() != _vector_size)
            return GrB_DIMENSION_MISMATCH;

        const M *mask_data = _mask->getDense()->get_vals();

        #pragma omp parallel for
        for(Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            if(mask_data[idx])
                _lambda_op(idx);
        }
    }
    else
    {
        #pragma omp parallel for
        for(Index i = 0; i < _nindexes; i++)
        {
            const Index idx = _indexes[i];
            _lambda_op(idx);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename LambdaOp, typename MonoidOpT>
LA_Info generic_dense_reduce_op(T *_tmp_val,
                                const Index _size,
                                LambdaOp &&_lambda_op,
                                MonoidOpT _monoid_op,
                                Descriptor *_desc)
{
    #pragma omp parallel
    {
        T local_res = _monoid_op.identity();
        #pragma omp for
        for(Index i = 0; i < _size; i++)
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

template <typename T, typename LambdaOp, typename MonoidOpT>
LA_Info generic_sparse_reduce_op(T *_tmp_val,
                                 const Index *_ids,
                                 const Index _nvals,
                                 LambdaOp &&_lambda_op,
                                 MonoidOpT _monoid_op,
                                 Descriptor *_desc)
{
    #pragma omp parallel
    {
        T local_res = _monoid_op.identity();
        #pragma omp for
        for(Index i = 0; i < _nvals; i++)
        {
            Index idx = _ids[i];
            local_res = _monoid_op(_lambda_op(idx), local_res);
        }

        #pragma omp critical
        {
            *_tmp_val = _monoid_op(*_tmp_val, local_res);
        };
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
