#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCSR : public MatrixContainer<T>
{
public:
    MatrixCSR();
    ~MatrixCSR();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    int target_socket;

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col);

    template<typename Y>
    friend void SpMV(MatrixCSR<Y> &_matrix,
                     MatrixCSR<Y> &_matrix_socket_dub,
                     DenseVector<Y> &_x,
                     DenseVector<Y> &_y,
                     Descriptor &_desc);

    template <typename Y>
    friend void SpMV(MatrixCSR<Y> &_matrix,
                     DenseVector<Y> &_x,
                     DenseVector<Y> &_y);

    void numa_aware_alloc();
};

#include "csr_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

