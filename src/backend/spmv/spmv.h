#pragma once

#include "spmv_seg.h"
#include "spmv_csr.h"
#include "spmv_coo.h"
#include "spmv_lav.h"
#include "spmv_sell_c.h"
#include "spmv_vect_csr.h"
#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T, typename SemiringT>
void SpMV(const Matrix<T> *_matrix,
          const Vector<T> *_x,
          Vector<T> *_y,
          Descriptor *_desc, 
          SemiringT _op)
{
    MatrixStorageFormat format;
    _matrix->get_format(&format);
    if(format == CSR)
        SpMV(((MatrixCSR<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == LAV)
        SpMV(((MatrixLAV<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == COO)
        SpMV(((MatrixCOO<T> *) _matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == CSR_SEG)
        SpMV(((MatrixSegmentedCSR<T> *)_matrix->get_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == SELL_C)
        SpMV(((MatrixSellC<T> *)_matrix->get_data()), _x->getDense(), _y->getDense(), _op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void VSpM(const Matrix<T> *_matrix,
          const Vector<T> *_x,
          Vector<T> *_y,
          Descriptor *_desc,
          SemiringT _op)
{
    MatrixStorageFormat format;
    _matrix->get_format(&format);
    if(format == CSR)
        SpMV(((MatrixCSR<T> *) _matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == LAV)
        SpMV(((MatrixLAV<T> *) _matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == COO)
        SpMV(((MatrixCOO<T> *) _matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == CSR_SEG)
        SpMV(((MatrixSegmentedCSR<T> *)_matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _op);
    else if(format == SELL_C)
        SpMV(((MatrixSellC<T> *)_matrix->get_transposed_data()), _x->getDense(), _y->getDense(), _op);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
