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
    if(_mask == NULL) // all active case
    {
        MatrixStorageFormat format;
        _matrix->get_format(&format);

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
                SpMV_all_active(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op);
            }
            #else
            SpMV_all_active(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op);
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
    }
    else
    {
        Desc_value mask_field;
        _desc->get(GrB_MASK, &mask_field);
        if (mask_field == GrB_SCMP)
        {
            throw "SCMP mask is not supported for now";
        }

        if(_mask->is_dense()) // dense case
        {
            SpMV_dense(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op, _mask->getDense());
        }
        else // sparse_case
        {
            SpMV_sparse(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op, _mask->getSparse());
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
    cout << "x ";
    _x->print_storage_type();
    cout << "y ";
    _y->print_storage_type();
    cout << "mask ";
    _mask->print_storage_type();

    if(_mask == NULL) // all active case
    {
        SpMV_all_active(_matrix->get_csc(), _x->getDense(), _y->getDense(), _accum, _op);
    }
    else
    {
        /*Desc_value mask_field;
        _desc->get(GrB_MASK, &mask_field);
        if (mask_field == GrB_SCMP)
        {
            throw "SCMP mask is not supported for now";
        }

        if(_mask->is_dense()) // dense case
        {
            SpMV_dense(_matrix->get_csc(), _x->getDense(), _y->getDense(), _accum, _op, _mask->getDense());
        }
        else // sparse_case
        {
            SpMV_sparse(_matrix->get_csc(), _x->getDense(), _y->getDense(), _accum, _op, _mask->getSparse());
        }*/
        SpMV_all_active(_matrix->get_csc(), _x->getDense(), _y->getDense(), _accum, _op);
    }

    if (_mask != NULL)
    {
        MatrixStorageFormat format;
        _matrix->get_format(&format);
        Desc_value mask_field;
        if (_mask != NULL) {
            _desc->get(GrB_MASK, &mask_field);
        }

        bool use_cmp;
        if (mask_field == GrB_SCMP)
        {
            use_cmp = true;
        }
        else
        {
            use_cmp = false;
        }
        cout << "USE CMP: " << use_cmp << endl;

        Y *result_vals = _y->getDense()->get_vals();
        const M *mask_vals = _mask->getDense()->get_vals();
        VNT mask_size = _mask->getDense()->get_size();
        #pragma omp parallel for
        for (VNT i = 0; i < mask_size; i++) {
            if (!(!use_cmp && (mask_vals[i]) || use_cmp && (!mask_vals[i])))
                result_vals[i] = 0;
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
