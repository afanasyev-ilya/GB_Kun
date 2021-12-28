#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
void SpMSpV(const Matrix<T> *_matrix,
            const Vector<T> *_x,
            Vector<T> *_y,
            Descriptor *_desc)
{
    MatrixStorageFormat format;
    _matrix->get_format(&format);

    //cout << "call" << endl;
    if(format == CSR) // CSR format (CSC format is inside CSR)
    {
        _matrix->print();
        SpMSpV_csr(((MatrixCSR<T> *) _matrix->get_transposed_data()), _x->getSparse(), _y->getSparse(), 2);
    }
    else {
        cout << "Unsupported format.";
    }
}

#include "spmspv_csr.h"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
