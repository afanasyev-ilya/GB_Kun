#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
class MatrixSellC : public MatrixContainer<T>
{
public:
    MatrixSellC();
    ~MatrixSellC();

    void build(VNT _nrows,
               VNT _ncols,
               ENT _nnz,
               const ENT *_row_ptr,
               const VNT *_col_ids,
               const T *_vals,
               int _target_socket = 0);

    void print() const;
    ENT get_nnz() const {return nnz;};
    void get_size(VNT* _size) const { *_size = size; };
private:
    int size, nnz;

    ENT *row_ptr;
    VNT *col_ids, *nnz_per_row;
    T *vals;
    VNT *rcmPerm, *rcmInvPerm;

    VNT C;
    VNT P;

    VNT nchunks, nnzSellC;
    VNT *chunkLen;
    ENT *chunkPtr;
    VNT *colSellC;
    T *valSellC;

    int unrollFac; //for kernel, just a work-around
    int nthreads;

    void NUMA_init();

    void construct_sell_c_sigma(VNT chunkHeight, VNT sigma, VNT pad = 1);
    void permute(VNT *perm, VNT* invPerm);

    T get(VNT _row, VNT _col) const;

    void print_stats();
    void print_connections(VNT _row);

    template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
    friend void SpMV(const MatrixSellC<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Workspace *_workspace);
};

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sell_c.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
