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
    void build(vector<vector<VNT>> &_tmp_csr_matrix, int _socket = 0) {};
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
    vector<pair<int, ENT>> sorted_segments;
    int load_balanced_threshold;

    void alloc(VNT _size, ENT _nnz);
    void free();

    template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
    friend void SpMV(const MatrixSegmentedCSR<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Workspace *_workspace);
};

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seg_csr_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

