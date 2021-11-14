#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/matrix_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Matrix
{
public:
    Matrix() {format = CSR;};
    ~Matrix() {delete data;};

    void build(VNT *_row_indices,
               VNT *_col_indices,
               T *_values,
               const VNT _size, // todo remove
               const ENT _nz);

    void set_preferred_format(MatrixStorageFormat _format) {format = _format;};
private:
    MatrixContainer<T> *data;

    MatrixStorageFormat format;

    template<typename Y>
    friend void SpMV(Matrix<Y> &_matrix,
                     Vector<Y> &_x,
                     Vector<Y> &_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(VNT *_row_indices,
                      VNT *_col_indices,
                      T *_values,
                      const VNT _size, // todo remove
                      const ENT _nz)
{
    if(format == CSR)
    {
        data = new MatrixCSR<T>;
    }
    else if(format == LAV)
    {
        data = new MatrixLAV<T>;
    }
    else if(format == COO)
    {
        data = new MatrixCOO<T>;
    }
    else if(format == CSR_SEG)
    {
        data = new MatrixSegmentedCSR<T>;
    }
    else
    {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    data->build(_row_indices, _col_indices, _values, _size, _nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
