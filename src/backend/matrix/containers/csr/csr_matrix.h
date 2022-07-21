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

    void deep_copy(const MatrixCSR<T> *_copy, int _target_socket = -1);

    void build(vector<vector<pair<VNT, T>>> &_tmp_csr, VNT _nrows, VNT _ncols);
    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _nrows, VNT _ncols, ENT _nnz);
    void build_from_csr_arrays(const ENT *_row_ptrs, const VNT *_col_ids, const T *_vals, VNT _nrows, VNT _ncols, ENT _nnz);

    void print() const;

    ENT get_nnz() const;
    VNT get_num_rows() const;
    VNT get_num_cols() const;

    ENT *get_row_ptr();
    const ENT *get_row_ptr() const;
    T *get_vals();
    const T *get_vals() const;
    VNT *get_col_ids();
    const VNT *get_col_ids() const;

    void set_row_ptr(ENT* ptr);

    VNT *get_rowdegrees();
    const VNT *get_rowdegrees() const;

    ENT get_degree(VNT _row);

    T get(VNT _row, VNT _col) const;

    const vector<pair<VNT, VNT>> &get_load_balancing_offsets() const;

    void resize(VNT _nrows, VNT _ncols, ENT _nnz);
    void numa_aware_realloc();

    bool is_symmetric();
    void to_symmetric();

    void calculate_degrees();

    void add_row(VNT _row_id);
    void remove_row(VNT _row_id);
    void add_val(VNT _row, VNT _col, T _val);
    void remove_val(VNT _row, VNT _col);
    // TODO do we need update edge here?

    void apply_modifications() const;
    void soft_apply_modifications() const;

    bool vertex_exists(VNT _row_id) const;

    bool has_unmerged_modifications() const { return ongoing_modifications; };
private:
    mutable VNT nrows, ncols;
    mutable ENT nnz;

    mutable ENT *row_ptr;
    mutable T *vals;
    mutable VNT *col_ids;
    mutable VNT *row_degrees;

    mutable bool ongoing_modifications;
    mutable ENT num_changes;
    mutable std::set<VNT> removed_vertices;
    mutable std::set<VNT> removed_rows;
    mutable std::set<VNT> restored_rows;
    mutable std::set<VNT> added_rows;
    mutable std::set<std::pair<VNT, ENT> > removed_edges;
    mutable std::map<VNT, std::map<std::pair<VNT, ENT>, T> > added_edges; /* supporting invariant that both adjacent
                                                                     vertices should have same edge in order for it
                                                                     to be valid when merging changes*/

    /* Vector of number_of_running_threads size
     * for i-th thread [i].first element means the beginning row to process
     * [i]. second means the first row not to be processed by i-th thread (right-non-inclusive intervals) */
    mutable vector<pair<VNT, VNT>> load_balancing_offsets;

    /* If load_balancing_offsets vector contains balanced offsets */
    mutable bool load_balancing_offsets_set;

    int target_socket;

    void alloc(VNT _nrows, VNT _ncols, ENT _nnz);
    void free();

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
    friend void SpMV_numa_aware(const MatrixCSR<A> *_matrix,
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

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_critical(const MatrixCSR<A> *_matrix,
                                         const SparseVector<X> *_x,
                                         DenseVector<Y> *_y,
                                         BinaryOpTAccum _accum,
                                         SemiringT _op,
                                         Descriptor *_desc,
                                         Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_critical_map(const MatrixCSR<A> *_matrix,
                                             const SparseVector<X> *_x,
                                             SparseVector<Y> *_y,
                                             BinaryOpTAccum _accum,
                                             SemiringT _op,
                                             Descriptor *_desc,
                                             Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_diff_vectors_tbb(const MatrixCSR<A> *_matrix,
                                                 const DenseVector<X> *_x,
                                                 DenseVector<Y> *_y,
                                                 BinaryOpTAccum _accum,
                                                 SemiringT op,
                                                 Descriptor *_desc,
                                                 Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void spmspv_unmasked_or_map(const MatrixCSR<A> *_matrix,
                                       const SparseVector<X> *_x,
                                       SparseVector<Y> *_y,
                                       BinaryOpTAccum _accum,
                                       SemiringT _op,
                                       Descriptor *_desc,
                                       Workspace *_workspace);

    template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMSpV_map_seq(const MatrixCSR<A> *_matrix,
                        const SparseVector <X> *_x,
                        SparseVector <Y> *_y,
                        Descriptor *_desc,
                        BinaryOpTAccum _accum,
                        SemiringT _op,
                        const Vector <M> *_mask);

    template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMSpV_map_par(const MatrixCSR<A> *_matrix,
                        const SparseVector <X> *_x,
                        SparseVector <Y> *_y,
                        Descriptor *_desc,
                        BinaryOpTAccum _accum,
                        SemiringT _op,
                        const Vector <M> *_mask);
};

#include "csr_matrix.hpp"
#include "build.hpp"
#include "symmetric.hpp"
#include "modification.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

