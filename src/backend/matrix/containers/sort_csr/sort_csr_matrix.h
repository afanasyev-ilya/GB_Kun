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

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _target_socket = 0);
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

    double *tmp_buffer;

    int target_socket;

    void alloc(VNT _size, ENT _nnz, int _target_socket);
    void free();
    void resize(VNT _size, ENT _nnz, int _target_socket);

    void construct_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz,
                       int _target_socket);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col) const;

    template <typename N, typename SemiringT>
    friend void SpMV(MatrixSortCSR<N> *_matrix,
                     const DenseVector<N> *_x,
                     DenseVector<N> *_y,
                     SemiringT op);
};

#include "sort_csr_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

