#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/csr/csr_matrix.h"
#include "containers/coo/coo_matrix.h"
#include "containers/seg_csr/seg_csr_matrix.h"
#include "containers/lav/lav_matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Matrix
{
public:
    Matrix() {};
    ~Matrix() {};

    void build(VNT *_row_indices,
               VNT *_col_indices,
               T *_values,
               const VNT _size, // todo remove
               const ENT _nz);

    void set_preferred_format(MatrixStorageFormat _format) {format = _format;};
private:
    MatrixCSR<T> csr_matrix; // for both sockets

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
    csr_matrix.import(_row_indices, _col_indices, _values, _size, _nz, 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
