#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
class MatrixSortCSR : public MatrixContainer<T>
{
public:
    MatrixSortCSR();
    ~MatrixSortCSR();

    void build(VNT *_row_degrees,
               VNT *_col_degrees,
               VNT _nrows,
               VNT _ncols,
               ENT _nnz,
               const ENT *_row_ptr,
               const VNT *_col_ids,
               const T *_vals,
               int _target_socket = 0);

    void print() const;

    ENT get_nnz() const {return nnz;};
    void get_size(VNT* _size) const {*_size = size;};

    ENT *get_row_ptr() {return row_ptr;};
    T *get_vals() {return vals;};
    VNT *get_col_ids() {return col_ids;};
private:
    VNT size;
    ENT nnz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    VNT *col_backward_conversion;

    double *tmp_buffer;

    int target_socket;

    void alloc(VNT _size, ENT _nnz, int _target_socket);
    void free();
    void resize(VNT _size, ENT _nnz, int _target_socket);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col) const;

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV(MatrixSortCSR<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op);
};

#include "sort_csr_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

