#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include  "subgraph_segment.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixSegmentedCSR : public MatrixContainer<T>
{
public:
    MatrixSegmentedCSR();
    ~MatrixSegmentedCSR();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print() {};
private:
    VNT size;
    ENT nz;

    VNT merge_blocks_number;

    int num_segments;

    SubgraphSegment<T> *subgraphs;

    void alloc(VNT _size, ENT _nz);
    void free();

    template<typename Y>
    friend void SpMV(MatrixSegmentedCSR<Y> &_matrix, DenseVector<Y> &_x, DenseVector<Y> &_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seg_csr_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
