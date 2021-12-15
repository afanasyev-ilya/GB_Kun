#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include  "subgraph_segment.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend {

template <typename T>
class MatrixSegmentedCSR;

template <typename T>
class MatrixSegmentedCSR : public MatrixContainer<T>
{
public:
    MatrixSegmentedCSR();
    ~MatrixSegmentedCSR();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket = 0);
    void print() const {};
    void get_size(VNT* _size) const {
        *_size = size;
    }

    ENT get_nnz() const {return nnz;};
private:
    VNT size;
    ENT nnz;

    VNT merge_blocks_number;

    int num_segments;

    SubgraphSegment<T> *subgraphs;

    void alloc(VNT _size, ENT _nnz);
    void free();

    template<typename Y, typename SemiringT>
    friend void SpMV(const MatrixSegmentedCSR<Y> *_matrix,
                     const DenseVector<Y> *_x,
                     DenseVector<Y> *_y, SemiringT op);
};

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seg_csr_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

