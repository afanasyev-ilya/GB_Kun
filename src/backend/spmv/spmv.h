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
          const DenseVector<T> *_x,
          DenseVector<T> *_y,
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
                                _x, _y, _accum, _op, _matrix->get_workspace());
            }
            else
            {
                SpMV_all_active(((MatrixCSR<T> *) _matrix->get_data()), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
            }
            #else
            SpMV_all_active(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _accum, _op, _desc,
                            _matrix->get_workspace());
            #endif
        }
        else if(format == LAV)
            SpMV(((MatrixLAV<T> *) _matrix->get_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        else if(format == COO)
            SpMV(((MatrixCOO<T> *) _matrix->get_data()), _x, _y, _op);
        else if(format == CSR_SEG)
            SpMV(((MatrixSegmentedCSR<T> *)_matrix->get_data()), _x, _y, _op);
        else if(format == SELL_C)
            SpMV(((MatrixSellC<T> *)_matrix->get_data()), _x, _y, _op);
        else if(format == SORTED_CSR)
            SpMV(((MatrixSortCSR<T> *)_matrix->get_data()), _x, _y, _op);
    }
    else
    {
        if(_mask->is_dense()) // dense case
        {
            SpMV_dense(_matrix->get_csr(), _x, _y, _accum, _op, _mask->getDense(), _desc, _matrix->get_workspace());
        }
        else // sparse_case
        {
            SpMV_sparse(_matrix->get_csr(), _x, _y, _accum, _op, _mask->getSparse(), _desc, _matrix->get_workspace());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void VSpM(const Matrix<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          Descriptor *_desc,
          BinaryOpTAccum _accum,
          SemiringT _op,
          const Vector<M> *_mask)
{
    if(_mask == NULL) // all active case
    {
        SpMV_all_active(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
    }
    else
    {
        if(_mask->is_dense()) // dense case
        {
            SpMV_dense(_matrix->get_csc(), _x, _y, _accum, _op, _mask->getDense(), _desc,
                       _matrix->get_workspace());
        }
        else // sparse_case
        {
            SpMV_sparse(_matrix->get_csc(), _x, _y, _accum, _op, _mask->getSparse(), _desc,
                        _matrix->get_workspace());
        }
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
