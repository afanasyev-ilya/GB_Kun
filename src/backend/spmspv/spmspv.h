#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
void SpMSpV(const Matrix<T> *_matrix,
            const SparseVector<T> *_x,
            Vector<T> *_y,
            Descriptor *_desc, int nb)
{
//    cout << "SPMSPV x: ";
//    _x->print_storage_type();
//    cout << "SPMSPV y: ";
//    _y->print_storage_type();
    SpMSpV_csr((MatrixCSR<T> *) _matrix->get_csc(), _x, _y->getDense(), nb);

//    cout << "SPMSPV result: ";
//    _y->force_to_dense();
//    _y->print();
}

#include "spmspv_csr.h"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
