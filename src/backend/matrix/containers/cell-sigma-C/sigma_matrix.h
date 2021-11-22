#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCellSigmaC : public MatrixContainer<T>
{
public:
    MatrixCellSigmaC();
    ~MatrixCellSigmaC();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
private:
    VNT size;
    ENT nz;

    // TODO
    int *group_numbers;

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    template <typename Y>
    friend void SpMV(MatrixCellSigmaC<Y> &_matrix,
                     DenseVector<Y> &_x,
                     DenseVector<Y> &_y);
};

#include "sigma_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

