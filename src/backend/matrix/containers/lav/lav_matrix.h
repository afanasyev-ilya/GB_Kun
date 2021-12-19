#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
struct LAVSegment
{

};

template <typename T>
class MatrixLAV : public MatrixContainer<T>
{
public:
    MatrixLAV();
    ~MatrixLAV();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket = 0);
    void print() const {};
    void get_size(VNT* _size) const {
        *_size = size;
    }

    ENT get_nnz() const {return nnz;};
private:
    VNT size;
    ENT nnz;

    VNT dense_segments;
    ENT **dense_row_ptr;
    VNT **dense_col_ids;
    T **dense_vals;
    VertexGroup *dense_vertex_groups;

    ENT *sparse_row_ptr;
    VNT *sparse_col_ids;
    T *sparse_vals;
    VertexGroup sparse_vertex_group;

    VNT *old_to_new;
    VNT *new_to_old;

    void alloc(VNT _size, ENT _nnz);
    void free();
    void resize(VNT _size, ENT _nnz);

    void construct_unsorted_csr(vector<vector<VNT>> &_tmp_col_ids,
                                vector<vector<T>> &_tmp_vals,
                                ENT **local_row_ptr,
                                VNT **local_col_ids,
                                T **local_vals,
                                VertexGroup *_vertex_group,
                                ENT _total_nnz);

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

