#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct CSRVertexGroupSellC
{
public:
    // init functions
    CSRVertexGroupSellC();
    ~CSRVertexGroupSellC();

    VNT get_min_nnz() const {return min_nnz;};
    VNT get_max_nnz() const {return max_nnz;};

    void print();

    /* import and preprocess API */
    void build(MatrixVectGroupCSR<T> *_matrix, VNT _bottom, VNT _top);
    void get_size(VNT* _size) {
        *_size = size;
    }
private:
    VNT *row_ids; // ids of vertices from this group
    VNT size; // size of this group

    VNT vector_segments_count; // similar to VE implementation
    ENT edges_count_in_ve;

    ENT *vector_group_ptrs;
    VNT *vector_group_sizes;
    VNT *vector_group_col_ids;
    T *vector_group_vals;

    VNT min_nnz, max_nnz;

    // helper functions
    bool id_in_range(VNT _src_id, VNT _nnz_count);

    template<typename Y>
    friend void SpMV_vector(const MatrixVectGroupCSR<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
