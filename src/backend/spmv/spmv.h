#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_coo.h"
#include "spmv_lav.h"
#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{

template <typename T>
void SpMV(const Matrix<T> *_matrix,
          const Vector<T> *_x,
          Vector<T> *_y,
          Descriptor *_desc)
          {
    MatrixStorageFormat format;
    _matrix->get_format(&format);
    if(format == CSR)
    {
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        SpMV(*((MatrixCSR<T> *) _matrix.data), *((MatrixCSR<T> *) _matrix.data_socket_dub), _x.dense, _y.dense, _desc);
        #else
        SpMV(((MatrixCSR<T> *) _matrix->get_Data()), _x->getDense(), _y->getDense());
        #endif
    }
    else if(format == LAV)
        SpMV(((MatrixLAV<T> *) _matrix->get_Data()), _x->getDense(), _y->getDense());
    else if(format == COO)
        SpMV(((MatrixCOO<T> *) _matrix->get_Data()), _x->getDense(), _y->getDense());
    else if(format == CSR_SEG)
        SpMV(((MatrixSegmentedCSR<T> *)_matrix->get_Data()), _x->getDense(), _y->getDense());
          }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
