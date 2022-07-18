#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_csr_neon.h"
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
    LOG_TRACE("Running SpMV")
    if(_mask == NULL) // all active case
    {
        MatrixStorageFormat format;
        _matrix->get_format(&format);
        Desc_value neon_mode;
        _desc->get(GrB_NEON, &neon_mode);
        if(format == CSR)
        {
            if(num_sockets_used() == 3) //TODO return to 1 after function fix
            {

                LOG_TRACE("Using NUMA-aware SPMV");

                SpMV_numa_aware(_matrix->get_csr(), _x, _y, _accum, _op, _matrix->get_workspace());
            }
            else
            {
                LOG_TRACE("Using single socket SPMV")
                if(_x == _y)
                {
                    SpMV_all_active_same_vectors(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                }
                else
                {
                    if (neon_mode == GrB_NEON_64) {
                        #ifdef __USE_KUNPENG__
                        SpMV_all_active_diff_vectors_neon(_matrix->get_csr(), _x, _y, _accum, _op, _desc,
                                                                _matrix->get_workspace());
                        #endif
                    } else if (neon_mode == GrB_NEON_32) {
                        #ifdef __USE_KUNPENG__
                        SpMV_all_active_diff_vectors_neon_short(_matrix->get_csr(), _x, _y, _accum, _op, _desc,
                                                                _matrix->get_workspace());
                        #endif
                    } else {
                        SpMV_all_active_diff_vectors(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                    }
                }
            }
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
    LOG_TRACE("VSpM")
    if(_mask == NULL) // all active case
    {
        MatrixStorageFormat format;
        _matrix->get_format(&format);
        if(format == CSR)
        {
            if(num_sockets_used() > 1)
            {
                LOG_TRACE("Using NUMA-aware VSpM")

                SpMV_numa_aware(_matrix->get_csc(), _x, _y, _accum, _op, _matrix->get_workspace());
            }
            else
            {
                LOG_TRACE("Using single socket VSpM")

                if(_x == _y)
                {
                    SpMV_all_active_same_vectors(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                }
                else
                {
                    SpMV_all_active_diff_vectors(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
                }
            }
        }
        else if(format == SELL_C)
        {
            SpMV(((MatrixSellC<A> *)_matrix->get_transposed_data()), _x, _y, _accum, _op, _matrix->get_workspace());
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
