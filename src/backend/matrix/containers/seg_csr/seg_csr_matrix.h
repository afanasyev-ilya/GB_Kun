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

    void get_size(VNT *_nz) const {
        *_nz = size;
    }
    void get_nz(ENT *_nz) const {
        *_nz = nz;
    }
    void get_segments(int *_nm) const {
        *_nm = num_segments;
    }

    void get_blocks(VNT *_nm) const {
        *_nm = merge_blocks_number;
    }
    SubgraphSegment<T> * get_segment(){
        return subgraphs;
    }
    const SubgraphSegment<T> * get_segment() const{
        return subgraphs;
    }

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
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
    friend void SpMV(const MatrixSegmentedCSR<Y> *_matrix,
                     const DenseVector<Y> *_x,
                     DenseVector<Y> *_y);
};

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "seg_csr_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

