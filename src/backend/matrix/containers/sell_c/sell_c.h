#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixSellC : public MatrixContainer<T>
{
public:
    MatrixSellC();
    ~MatrixSellC();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print() {};
private:
    int size, nz;

    int *rowPtr, *col, *nnzPerRow;
    T *val;
    int *rcmPerm, *rcmInvPerm;

    int C;
    int P;

    int nchunks, nnzSellC;
    int *chunkLen;
    int *chunkPtr;
    int *colSellC;
    double *valSellC;

    int unrollFac; //for kernel, just a work-around
    int nthreads;

    void NUMA_init();

    void constructSellCSigma(int C, int sigma, int P=1);
    void permute(int *perm, int*  invPerm);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sell_c.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
