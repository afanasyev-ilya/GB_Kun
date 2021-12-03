#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
class MatrixLAV : public MatrixContainer<T>
{
public:
    MatrixLAV();
    ~MatrixLAV();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
    void get_size(VNT* _size) {
        *_size = size;
    }

    ENT get_nnz() {return nz;};
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

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col);

    void prepare_hub_data(map<VNT, ENT> &_freqs);

    template<typename Y, typename SemiringT>
    friend void SpMV(const MatrixLAV<Y> *_matrix,
                     const DenseVector<Y> *_x,
                     DenseVector<Y> *_y, SemiringT op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lav_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

