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

template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMV(const Matrix<A> *_matrix,
          const DenseVector<X> *_x,
          DenseVector<Y> *_y,
          Descriptor *_desc,
          BinaryOpTAccum _accum,
          SemiringT _op,
          const Vector<M> *_mask)
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
                SpMV_numa_aware(((MatrixCSR<A> *) _matrix->get_csr()), ((MatrixCSR<A> *) _matrix->get_data_dub()),
                                _x, _y, _accum, _op, _matrix->get_workspace());
            }
            else
            {
                if(_x == _y)
                {
                    SpMV_all_active_same_vectors(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                }
                else
                {
                    SpMV_all_active_diff_vectors(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                }
            }
            #else
            SpMV_all_active(((MatrixCSR<T> *) _matrix->get_csr()), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
            #endif
        }
        else if(format == LAV)
            SpMV(((MatrixLAV<A> *) _matrix->get_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        else if(format == COO)
            SpMV(((MatrixCOO<A> *) _matrix->get_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        else if(format == CSR_SEG)
            SpMV(((MatrixSegmentedCSR<A> *)_matrix->get_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        else if(format == SELL_C)
            SpMV(((MatrixSellC<A> *)_matrix->get_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        else if(format == SORTED_CSR)
            SpMV(((MatrixSortCSR<A> *)_matrix->get_data()), _x, _y, _accum, _op);
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
        MatrixStorageFormat format;
        _matrix->get_format(&format);
        if(format == CSR)
        {
            if(_x == _y)
            {
                SpMV_all_active_same_vectors(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
            }
            else
            {
                SpMV_all_active_diff_vectors(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
            }
        }
        else if(format == SELL_C)
        {
            cout << "doing check" << endl;
            DenseVector<Y> check(_matrix->get_nrows());
            for(int i = 0; i < _matrix->get_nrows(); i++)
                check.get_vals()[i] = _y->get_vals()[i];


            SpMV(((MatrixSellC<A> *)_matrix->get_transposed_data()), _x, _y, _accum, _op, _matrix->get_workspace());


            SpMV_all_active_diff_vectors(_matrix->get_csc(), _x, &check, _accum, _op, _desc, _matrix->get_workspace());
            if(check == (*_y)) {
                cout << "well its ok" << endl;
            }
            else {
                cout << "problem occured!" << endl;
            }
        }
        else if(format == CSR_SEG)
        {
            SpMV(((MatrixSegmentedCSR<A> *)_matrix->get_transposed_data()), _x, _y, _accum, _op, _matrix->get_workspace());
        }
        else
        {
            throw "unsupported matrix storage format in VSpM";
        }
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
