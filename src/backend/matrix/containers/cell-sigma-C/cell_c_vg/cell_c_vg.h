#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct CSRVertexGroupCellC
{
public:
    // init functions
    CSRVertexGroupCellC();
    ~CSRVertexGroupCellC();

    VNT get_min_nz() const {return min_nz;};
    VNT get_max_nz() const {return max_nz;};

    void print();

    /* import and preprocess API */
    void build(MatrixCellSigmaC<T> *_matrix, VNT _bottom, VNT _top);
private:
    VNT *vertex_ids; // ids of vertices from this group
    VNT size; // size of this group

    VNT vector_segments_count; // similar to VE implementation
    ENT edges_count_in_ve;

    ENT *vector_group_ptrs;
    VNT *vector_group_sizes;
    VNT *vector_group_col_ids;
    T *vector_group_vals;

    VNT min_nz, max_nz;

    // helper functions
    bool id_in_range(VNT _src_id, VNT _nz_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
