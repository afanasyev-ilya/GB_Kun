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

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();
    ENT get_nnz() {return nz;};
private:
    int size, nz;

    ENT *row_ptr;
    VNT *col_ids, *nz_per_row;
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

    T get(VNT _row, VNT _col);

    template <typename Y>
    friend void SpMV(const MatrixSellC<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);

    void generateHPCG(int nx, int ny, int nz);
};

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sell_c.hpp"
#include "build.hpp"
#include "hpcg.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
