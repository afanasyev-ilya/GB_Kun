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

    ENT get_nnz() const {return this->nnz;};
    VNT get_num_rows() const {return nrows;};
    VNT get_num_cols() const {return ncols;};

    ENT *get_row_ptr() {return row_ptr;};
    const ENT *get_row_ptr() const {return row_ptr;};
    T *get_vals() {return vals;};
    const T *get_vals() const {return vals;};
    VNT *get_col_ids() {return col_ids;};
    const VNT *get_col_ids() const {return col_ids;};

    void set_row_ptr(ENT* ptr) {
        row_ptr = ptr;
    }

    VNT *get_rowdegrees() {return row_degrees;};
    const VNT *get_rowdegrees() const {return row_degrees;};

    ENT get_degree(VNT _row) {return row_ptr[_row + 1] - row_ptr[_row];};

    T get(VNT _row, VNT _col) const;

    const vector<pair<VNT, VNT>> &get_load_balancing_offsets() const;

    void resize(VNT _nrows, VNT _ncols, ENT _nnz);
    void numa_aware_realloc();

    bool is_symmetric();
    void to_symmetric();


    void calculate_degrees();
private:
    VNT nrows, ncols;
    ENT nnz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;
    VNT *row_degrees;

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

#ifdef __USE_KUNPENG__

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_diff_vectors_neon(const MatrixCSR<A> *_matrix,
                                             const DenseVector <X> *_x,
                                             DenseVector <Y> *_y,
                                             BinaryOpTAccum _accum,
                                             SemiringT op,
                                             Descriptor *_desc,
                                             Workspace *_workspace);

    template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
    friend void SpMV_all_active_diff_vectors_neon_short(const MatrixCSR<A> *_matrix,
                                                  const DenseVector <X> *_x,
                                                  DenseVector <Y> *_y,
                                                  BinaryOpTAccum _accum,
                                                  SemiringT op,
                                                  Descriptor *_desc,
                                                  Workspace *_workspace);

#endif

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

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

