#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {


class VertexGroup
{
public:
    bool in_range(ENT _connections_num) const
    {
        if(_connections_num >= min_threshold && _connections_num < max_threshold)
            return true;
        else
            return false;
    }

    void set_thresholds(ENT _min_threshold, ENT _max_threshold)
    {
        min_threshold = _min_threshold;
        max_threshold = _max_threshold;
        data.clear();
    }

    void push_back(VNT _row)
    {
        data.push_back(_row);
    }

    VNT get_size() const
    {
        return size;
    }

    const VNT *get_data() const
    {
        return opt_data;
    }

    void finalize_creation(int _target_socket)
    {
        size = (VNT)data.size();
        MemoryAPI::numa_aware_alloc(&opt_data, size, _target_socket);
        MemoryAPI::copy(opt_data, &data[0], size);
    }
private:
    ENT min_threshold;
    ENT max_threshold;

    std::vector<VNT> data;
    VNT *opt_data;
    VNT size;
};

template <typename T>
class MatrixCSR : public MatrixContainer<T>
{
public:
    MatrixCSR();
    ~MatrixCSR();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _target_socket = 0);
    void print() const;

    ENT get_nnz() const {return nnz;};
    void get_size(VNT* _size) const {*_size = size;};

    ENT *get_row_ptr() {return row_ptr;};
    T *get_vals() {return vals;};
    VNT *get_col_ids() {return col_ids;};
private:
    VNT size;
    ENT nnz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    static const int vg_num = 8; // 9 is best currently
    VertexGroup vertex_groups[vg_num];

    double *tmp_buffer;

    int target_socket;

    void alloc(VNT _size, ENT _nnz, int _target_socket);
    void free();
    void resize(VNT _size, ENT _nnz, int _target_socket);

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz,
                                int _target_socket);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col) const;

    template <typename N, typename SemiringT>
    friend void SpMV(const MatrixCSR<N> *_matrix,
              const DenseVector<N> *_x,
              DenseVector<N> *_y, SemiringT op);

    template <typename N, typename SemiringT>
    friend void SpMV_numa_aware(MatrixCSR<N> *_matrix,
                                MatrixCSR<N> *_matrix_socket_dub,
                                const DenseVector<N> *_x,
                                DenseVector<N> *_y,
                                SemiringT op);

    template <typename N, typename SemiringT>
    friend void SpMV_non_optimized(MatrixCSR<N> *_matrix,
                                   const DenseVector<N> *_x,
                                   DenseVector<N> *_y,
                                   SemiringT op);

    template <typename N, typename SemiringT>
    friend void SpMV_test(const MatrixCSR<N> *_matrix,
                   const DenseVector<N> *_x,
                   DenseVector<N> *_y, SemiringT op);

    void prepare_vg_lists(int _target_socket);
};

#include "csr_matrix.hpp"
#include "build.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

