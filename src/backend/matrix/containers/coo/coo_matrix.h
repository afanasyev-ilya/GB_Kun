#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas {
namespace backend {

template <typename T>
class MatrixCOO : public MatrixContainer<T>
{
public:
    MatrixCOO();
    ~MatrixCOO();

    void build(VNT _num_rows, ENT _nnz, const ENT *_row_ptr, const VNT *_col_ids, const T *_vals, int _socket = 0);

    void print() const;
    void get_size(VNT* _size) const {*_size = size;}

    ENT get_nnz() const {return nnz;};
private:
    VNT size;
    ENT nnz;

    VNT *row_ids;
    VNT *col_ids;
    T *vals;

    ENT *thread_bottom_border;
    ENT *thread_top_border;

    void alloc(VNT _size, ENT _nnz);
    void free();

    void resize(VNT _size, ENT _nnz);

    template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
    friend void SpMV(const MatrixCOO<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT _op,
                     Workspace *_workspace);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "coo_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

