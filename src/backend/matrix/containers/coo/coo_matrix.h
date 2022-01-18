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

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket = 0);
    void build(vector<vector<VNT>> &_tmp_csr_matrix, int _socket = 0) {};

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

    template <typename Y, typename SemiringT>
    friend void SpMV(const MatrixCOO<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y, SemiringT op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "coo_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

