#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

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
        throw "Error in SpVSpM : unsupported additive operation in semiring";
    }

    if (_mask != 0)
    {
        apply_mask(_y, old_y_vals, _desc, _accum, _mask, _matrix->get_workspace());
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bucket {
    VNT row;
    T val;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpV(const Matrix<T> *_matrix,
            const SparseVector<T> *_x,
            Vector<T> *_y,
            Descriptor *_desc,
            int nb)
{
    VNT matrix_size, nz;
    _matrix->get_csc()->get_size(&matrix_size);
    _x->get_nnz(&nz);
    //long long max_number_of_insertions = nz*_matrix->get_csc()->get_max_degree();
    long long max_number_of_insertions = nz*matrix_size;

    int nt = 32;
    //auto v = _matrix->get_workspace()->get_spmspv_buffer();
    auto v = (char *)calloc(1, sizeof(int) * (2*nb + nt * nb) + sizeof(float) * (matrix_size) + sizeof(bucket<T>) * (nb * max_number_of_insertions));
    int memory_offset = 0;
    auto bucket_amount = (int *)(v + memory_offset);
    memory_offset += nb * sizeof(int);
    auto offset_ = (int *)(v + memory_offset);
    memory_offset += nb * nt * sizeof(int);
    auto *SPA = (float *)(v + memory_offset);
    memory_offset += matrix_size * sizeof(float);
    int *offset = (int *)(v + memory_offset);
    memory_offset += nb * sizeof(int);
    auto buckets = (bucket<T> *)(v + memory_offset);
    //memory_offset += nb * max_number_of_insertions * sizeof(bucket<T>);

    SpMSpV_csr((MatrixCSR<T> *) _matrix->get_csc(), _x, _y->getDense(), (int)nb, (int)nt,
               (int *)bucket_amount, (int *)offset_, (bucket<T> *)buckets, (float*)SPA, (int*)offset);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpV_alloc(const Matrix<T> *_matrix,
                  const SparseVector<T> *_x,
                  Vector<T> *_y,
                  Descriptor *_desc,
                  int nb)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "spmspv_buckets.h"
#include "spmspv_csr.h"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
