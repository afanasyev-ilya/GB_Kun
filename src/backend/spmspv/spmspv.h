#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

#include "spmspv_buckets.h"
#include "spmspv_atomics.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MASK_TRUE 1
#define MASK_FALSE 0

template <typename Y, typename M, typename BinaryOpTAccum>
void apply_mask(DenseVector <Y> *_y,
                Y *_old_y_vals,
                Descriptor *_desc,
                BinaryOpTAccum _accum,
                const Vector <M> *_mask,
                Workspace *_workspace)
{
    Desc_value mask_field;
    _desc->get(GrB_MASK, &mask_field);
    Y *y_vals = _y->get_vals();

    if (mask_field != GrB_STR_COMP)
    {
        if(_mask->is_dense())
        {
            const M *mask_vals = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for (VNT i = 0; i < _mask->get_size(); i++)
            {
                if(mask_vals[i] != 0)
                    y_vals[i] = _accum(_old_y_vals[i], y_vals[i]);
                else
                    y_vals[i] = 0;
            }
        }
        else
        {
            const VNT mask_nvals = _mask->getSparse()->get_nvals();
            const VNT *mask_ids = _mask->getSparse()->get_ids();
            #pragma omp parallel for
            for (VNT idx = 0; idx < mask_nvals; idx++)
            {
                VNT i = mask_ids[idx];
                y_vals[i] = _accum(_old_y_vals[i], y_vals[i]); // TODO problem
            }
        }
    }
    else
    {
        if(_mask->is_dense())
        {
            const M *mask_vals = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for (VNT i = 0; i < _mask->get_size(); i++)
            {
                if(mask_vals[i] == 0) // == 0 since CMP mask
                    y_vals[i] = _accum(_old_y_vals[i], y_vals[i]);
                else
                    y_vals[i] = 0;
            }
        }
        else
        {
            bool *dense_mask = (bool*)_workspace->get_mask_conversion();

            const VNT mask_nvals = _mask->getSparse()->get_nvals();
            const VNT *mask_ids = _mask->getSparse()->get_ids();
            #pragma omp parallel
            {
                #pragma omp for
                for (VNT i = 0; i < _mask->get_size(); i++)
                    dense_mask[i] = MASK_TRUE;

                #pragma omp for
                for (VNT i = 0; i < mask_nvals; i++)
                {
                    VNT mask_id = mask_ids[i];
                    dense_mask[mask_id] = MASK_FALSE; // we deactivate all values from original mask since this is CMP
                }

                #pragma omp for
                for (VNT i = 0; i < _mask->get_size(); i++)
                {
                    if(dense_mask[i] == MASK_TRUE)
                        y_vals[i] = _accum(_old_y_vals[i], y_vals[i]);
                    else
                        y_vals[i] = 0;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV(const Matrix<A> *_matrix,
            bool _transposed_matrix,
            const SparseVector <X> *_x,
            DenseVector <Y> *_y,
            Descriptor *_desc,
            BinaryOpTAccum _accum,
            SemiringT _op,
            const Vector <M> *_mask)
{
    auto add_op = extractAdd(_op);
    Y *y_vals = _y->get_vals();
    Y *old_y_vals = (Y*)_matrix->get_workspace()->get_shared_one();
    memcpy(old_y_vals, y_vals, sizeof(Y)*_y->get_size());
    /*!
      * /brief atomicAdd() 3+5  = 8
      *        atomicSub() 3-5  =-2
      *        atomicMin() 3,5  = 3
      *        atomicMax() 3,5  = 5
      *        atomicOr()  3||5 = 1
      *        atomicXor() 3^^5 = 0
    */
    int functor = add_op(3, 5);
    if (functor == 8)
    {
        if(!_transposed_matrix)
            spmspv_unmasked_add(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
        else
            spmspv_unmasked_add(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
    }
    else if (functor == 1)
    {
        if(!_transposed_matrix)
            spmspv_unmasked_or(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
        else
            spmspv_unmasked_or(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
    }
    else
    {
        if(!_transposed_matrix)
            spmspv_unmasked_critical(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
        else
            spmspv_unmasked_critical(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
    }

    /*int nb = 10;
    if(!_transposed_matrix)
        spmspv_buckets(_matrix->get_csc(), _x, _y, (int)nb, _matrix->get_workspace(), _accum, _op);
    else
        spmspv_buckets(_matrix->get_csr(), _x, _y, (int)nb, _matrix->get_workspace(), _accum, _op);*/

    if (_mask != 0)
    {
        apply_mask(_y, old_y_vals, _desc, _accum, _mask, _matrix->get_workspace());
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
