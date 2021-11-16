#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCOO : public MatrixContainer<T>
{
public:
    MatrixCOO();
    ~MatrixCOO();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
private:
    VNT size;
    ENT nz;

    VNT *row_ids;
    VNT *col_ids;
    T *vals;

    void alloc(VNT _size, ENT _nz);
    void free();

    void resize(VNT _size, ENT _nz);

    template<typename Y>
    friend void SpMV(MatrixCOO<Y> &_matrix, DenseVector<Y> &_x, DenseVector<Y> &_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "coo_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

