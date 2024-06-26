#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
struct LAVSegment
{
    ENT *row_ptr;
    VNT *col_ids;
    T *vals;

    VertexGroup vertex_list;

    static const int vg_num = 9;
    VertexGroup vertex_groups[vg_num];
    ENT nnz;
    VNT size;

    VNT min_col_id;
    VNT max_col_id;

    void free()
    {
        delete []row_ptr;
        delete []col_ids;
        delete []vals;
    }
};

template <typename T>
class MatrixLAV : public MatrixContainer<T>
{
public:
    MatrixLAV();
    ~MatrixLAV();

    void build(VNT *_row_degrees,
               VNT *_col_degrees,
               VNT _nrows,
               VNT _ncols,
               ENT _nnz,
               const ENT *_row_ptr,
               const VNT *_col_ids,
               const T *_vals,
               int _target_socket = 0);

    void print() const {};
    void get_size(VNT* _size) const
    {
        *_size = nrows;
    }

    ENT get_nnz() const {return nnz;};
private:
    VNT nrows, ncols;
    ENT nnz;

    VNT dense_segments_num;
    LAVSegment<T> *dense_segments;
    LAVSegment<T> sparse_segment;

    VNT *col_new_to_old;
    VNT *col_old_to_new;

    void alloc(VNT _size, ENT _nnz);
    void free();
    void resize(VNT _size, ENT _nnz);

    void construct_unsorted_csr(vector<vector<VNT>> &_tmp_col_ids,
                                vector<vector<T>> &_tmp_vals,
                                LAVSegment<T> *_cur_segment,
                                ENT _total_nnz);

    template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
    friend void SpMV(const MatrixLAV<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Workspace *_workspace);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lav_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

