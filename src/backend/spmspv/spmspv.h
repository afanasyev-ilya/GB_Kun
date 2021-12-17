#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpV(Matrix<T> &_matrix,
          Vector<T> &_x,
          Vector<T> &_y,
          Descriptor &_desc)
{
    if(_matrix.format == CSR) // CSR format (CSC format is inside CSR)
    {
        SpMV(*((MatrixCSR<T> *) _matrix.data), _x.sparse, _y.sparse);
    }
    else {
        cout << "Unsupported format."
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
