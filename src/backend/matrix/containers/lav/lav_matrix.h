#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define HUB_VERTICES 131072

template <typename T>
class MatrixLAV : public MatrixContainer<T>
{
public:
    MatrixLAV();
    ~MatrixLAV();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    VNT *hub_conversion_array;

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(int _row, int _col);
    T get(int _row, int _col);

    void prepare_hub_data(map<int, int> &_freqs);

    template<typename Y>
    friend void SpMV(MatrixLAV<Y> &_matrix, DenseVector<Y> &_x, DenseVector<Y> &_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lav_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

