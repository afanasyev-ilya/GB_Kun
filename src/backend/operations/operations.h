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
        // TODO if mask is sparse

        if(_mask->getDense()->get_size() != _size)
            return GrB_DIMENSION_MISMATCH;

        const M *mask_data = _mask->getDense()->get_vals();

        #pragma omp parallel for
        for(Index i = 0; i < _size; i++)
        {
            if(mask_data[i])
                _lambda_op(i);
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

}
}