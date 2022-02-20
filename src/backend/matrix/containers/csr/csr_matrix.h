#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vertex_group.h"

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCSR : public MatrixContainer<T>
{
public:
    MatrixCSR();
    ~MatrixCSR();

    void deep_copy(MatrixCSR<T> *_copy, int _target_socket = -1);

    void build(vector<vector<pair<VNT, T>>> &_tmp_csr, VNT _nrows, VNT _ncols, int _target_socket);
    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _nrows, VNT _ncols,
               ENT _nnz, int _target_socket = 0);

    void print() const;

    ENT get_nnz() const {return nnz;};
    VNT get_num_rows() const {return nrows;};
    VNT get_num_cols() const {return ncols;};

    ENT *get_row_ptr() {return row_ptr;};
    const ENT *get_row_ptr() const {return row_ptr;};
    T *get_vals() {return vals;};
    const T *get_vals() const {return vals;};
    VNT *get_col_ids() {return col_ids;};
    const VNT *get_col_ids() const {return col_ids;};

    VNT *get_rowdegrees() {return row_degrees;};
    const VNT *get_rowdegrees() const {return row_degrees;};

    bool can_use_static_balancing() const {return static_ok_to_use;};
    ENT get_degree(VNT _row) {return row_ptr[_row + 1] - row_ptr[_row];};

    T get(VNT _row, VNT _col) const;

private:
    VNT nrows, ncols;
    ENT nnz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;
    VNT *row_degrees;

    static const int vg_num = 9; // 9 is best currently
    VertexGroup vertex_groups[vg_num];
    bool static_ok_to_use;

    int target_socket;

    void alloc(VNT _nrows, VNT _ncols, ENT _nnz, int _target_socket);
    void free();
    void resize(VNT _nrows, VNT _ncols, ENT _nnz, int _target_socket);

    bool is_non_zero(VNT _row, VNT _col);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_same_vectors(const MatrixCSR<A> *_matrix,
                                             const DenseVector <X> *_x,
                                             DenseVector <Y> *_y,
                                             BinaryOpTAccum _accum,
                                             SemiringT op,
                                             Descriptor *_desc,
                                             Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_diff_vectors(const MatrixCSR<A> *_matrix,
                                             const DenseVector <X> *_x,
                                             DenseVector <Y> *_y,
                                             BinaryOpTAccum _accum,
                                             SemiringT op,
                                             Descriptor *_desc,
                                             Workspace *_workspace);

    template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_dense(const MatrixCSR<A> *_matrix,
                           const DenseVector<X> *_x,
                           DenseVector<Y> *_y,
                           BinaryOpTAccum _accum,
                           SemiringT op,
                           const DenseVector<M> *_mask,
                           Descriptor *_desc,
                           Workspace *_workspace);

    template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_sparse(const MatrixCSR<A> *_matrix,
                            const DenseVector<X> *_x,
                            DenseVector<Y> *_y,
                            BinaryOpTAccum _accum,
                            SemiringT op,
                            const SparseVector<M> *_mask,
                            Descriptor *_desc,
                            Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_numa_aware(MatrixCSR<A> *_matrix,
                                MatrixCSR<A> *_matrix_socket_dub,
                                const DenseVector<X> *_x,
                                DenseVector<Y> *_y,
                                BinaryOpTAccum _accum,
                                SemiringT op,
                                Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_static(const MatrixCSR<A> *_matrix,
                                       const DenseVector<X> *_x,
                                       DenseVector<Y> *_y,
                                       BinaryOpTAccum _accum,
                                       SemiringT op,
                                       Descriptor *_desc,
                                       Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_add(const MatrixCSR<A> *_matrix,
                                    const SparseVector<X> *_x,
                                    DenseVector<Y> *_y,
                                    BinaryOpTAccum _accum,
                                    SemiringT op,
                                    Descriptor *_desc,
                                    Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_add_opt(const MatrixCSR<A> *_matrix,
                                        const SparseVector<X> *_x,
                                        DenseVector<Y> *_y,
                                        BinaryOpTAccum _accum,
                                        SemiringT _op,
                                        Descriptor *_desc,
                                        Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_or(const MatrixCSR<A> *_matrix,
                                   const SparseVector<X> *_x,
                                   DenseVector<Y> *_y,
                                   BinaryOpTAccum _accum,
                                   SemiringT _op,
                                   Descriptor *_desc,
                                   Workspace *_workspace);

    void prepare_vg_lists(int _target_socket);
    void numa_aware_realloc();
    void check_if_static_can_be_used();
    void calculate_degrees();
};

#include "csr_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

