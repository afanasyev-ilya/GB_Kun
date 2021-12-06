#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
class MatrixCSR : public MatrixContainer<T>
{
public:
    MatrixCSR();
    ~MatrixCSR();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print() const;

    ENT get_nnz() const {return nz;};
    void get_size(VNT* _size) const {*_size = size;};
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

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col) const;

    template <bool CMP, typename N, typename Y, typename SemiringT>
    friend void SpMV(const MatrixCSR<N> *_matrix,
              const DenseVector<N> *_x,
              DenseVector<N> *_y, SemiringT op, const Vector<Y> *_mask);

    void numa_aware_alloc();
};

#include "csr_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

