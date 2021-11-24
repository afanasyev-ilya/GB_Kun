#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_coo.h"
#include "spmv_lav.h"
#include "spmv_cell_sigma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(Matrix<T> &_matrix,
          Vector<T> &_x,
          Vector<T> &_y,
          Descriptor &_desc)
{
    if(_matrix.format == CSR)
    {
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        SpMV(*((MatrixCSR<T> *) _matrix.data), *((MatrixCSR<T> *) _matrix.data_socket_dub), _x.dense, _y.dense, _desc);
        #else
        SpMV(*((MatrixCSR<T> *) _matrix.data), _x.dense, _y.dense);
        #endif
    }
    else if(_matrix.format == LAV)
        SpMV(*((MatrixLAV<T> *) _matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == COO)
        SpMV(*((MatrixCOO<T> *) _matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == CSR_SEG)
        SpMV(*((MatrixSegmentedCSR<T> *)_matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == CELL_SIGMA_C)
        SpMV(*((MatrixCellSigmaC<T> *)_matrix.data), _x.dense, _y.dense);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
