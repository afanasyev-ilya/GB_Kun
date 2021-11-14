#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_coo.h"
#include "spmv_lav.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(Matrix<T> &_matrix,
          Vector<T> &_x,
          Vector<T> &_y)
{
    if(_matrix.format == CSR)
        SpMV(*((MatrixCSR<T>*)_matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == LAV)
        SpMV(*((MatrixLAV<T> *) _matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == COO)
        SpMV(*((MatrixCOO<T> *) _matrix.data), _x.dense, _y.dense);
    else if(_matrix.format == CSR_SEG)
        SpMV(*((MatrixSegmentedCSR<T> *)_matrix.data), _x.dense, _y.dense);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
