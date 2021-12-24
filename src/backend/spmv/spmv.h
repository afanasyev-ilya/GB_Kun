#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_coo.h"
#include "spmv_lav.h"
#include "spmv_sell_c.h"
#include "spmv_vect_csr.h"
#include "spmv_sort_csr.h"
#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T, typename Y, typename SemiringT, typename BinaryOpTAccum>
void SpMV(const Matrix<T> *_matrix,
          const Vector<T> *_x,
          Vector<T> *_y,
          Descriptor *_desc,
          BinaryOpTAccum _accum,
          SemiringT _op,
          const Vector<Y> *_mask)
{
    MatrixStorageFormat format;
    _matrix->get_format(&format);
    Desc_value mask_field;
    if (_mask != NULL) {
        _desc->get(GrB_MASK, &mask_field);
    }
    auto zero_val = _op.identity();

    if(format == CSR)
    {
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(omp_get_max_threads() == THREADS_PER_SOCKET*2)
        {
            SpMV_numa_aware(((MatrixCSR<T> *) _matrix->get_data()), ((MatrixCSR<T> *) _matrix->get_data_dub()),
                            _x->getDense(), _y->getDense(), _accum, _op);
        }
        else
        {
            SpMV(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op);
        }
        #else
        SpMV(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op);
        #endif
    }
    else if(format == LAV)
        SpMV(((MatrixLAV<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == COO)
        SpMV(((MatrixCOO<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == CSR_SEG)
        SpMV(((MatrixSegmentedCSR<T> *)_matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == SELL_C)
        SpMV(((MatrixSellC<T> *)_matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == SORTED_CSR)
        SpMV(((MatrixSortCSR<T> *)_matrix->get_data()), _x->getDense(), _y->getDense(), _op);

    if (_mask != NULL)
    {
        bool use_cmp;
        if (mask_field == GrB_SCMP)
        {
            use_cmp = true;
        }
        else
        {
            use_cmp = false;
        }
        T *result_vals = _y->getDense()->get_vals();
        const Y *mask_vals = _mask->getDense()->get_vals();
        VNT mask_size = _mask->getDense()->get_size();
        #pragma omp parallel for
        for (VNT i = 0; i < mask_size; i++) {
            if (mask_vals[i] && !use_cmp)
                result_vals[i] = zero_val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void VSpM(const Matrix<A> *_matrix,
          const Vector<X> *_x,
          Vector<Y> *_y,
          Descriptor *_desc,
          BinaryOpTAccum _accum,
          SemiringT _op,
          const Vector<M> *_mask)
{
    MatrixStorageFormat format;
    _matrix->get_format(&format);
    auto zero_val = _op.identity();

    Desc_value mask_field;
    if (_mask != NULL) {
        _desc->get(GrB_MASK, &mask_field);
    }

    if(format == CSR)
    {
        SpMV(((MatrixCSR<A> *) _matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _accum, _op);
    }

    if (_mask != NULL)
    {
        bool use_cmp;
        if (mask_field == GrB_SCMP)
        {
            use_cmp = true;
        }
        else
        {
            use_cmp = false;
        }
        Y *result_vals = _y->getDense()->get_vals();
        const M *mask_vals = _mask->getDense()->get_vals();
        VNT mask_size = _mask->getDense()->get_size();
        #pragma omp parallel for
        for (VNT i = 0; i < mask_size; i++) {
            if (mask_vals[i] && !use_cmp)
                result_vals[i] = zero_val;
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
