#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    template <typename T>
    void build(MatrixCellSigmaC<T> *_matrix, VNT _bottom, VNT _top);
private:
    VNT *vertex_ids; // ids of vertices from this group
    VNT size; // size of this group

    VNT vector_segments_count; // similar to VE implementation
    ENT edges_count_in_ve;

    ENT *vector_group_ptrs;
    VNT *vector_group_sizes;
    VNT *vector_group_adjacent_ids;

    ENT *old_edge_indexes; // edge IDS required for conversion (take + O|V| space)

    VNT min_nz, max_nz;

    // helper functions
    bool id_in_range(VNT _src_id, VNT _nz_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD_CSR_VERTEX_GROUP_CELL_C_DATA(vertex_group)  \
VNT *vertex_ids = _group_data.get_vertex_ids(); \
VNT size = _group_data.get_size(); \
\
VNT vector_segments_count = _group_data.get_vector_segments_count(); \
ENT edges_count_in_ve = _group_data.get_edges_count_in_ve();\
\
ENT *vector_group_ptrs = _group_data.get_vector_group_ptrs(); \
VNT *vector_group_sizes = _group_data.get_vector_group_sizes();\
VNT *vector_group_adjacent_ids = _group_data.get_vector_group_adjacent_ids();\
\
ENT *old_edge_indexes = _group_data.get_old_edge_indexes();\
\
VNT min_nz = _group_data.get_min_nz();\
VNT max_nz = _group_data.get_max_nz();\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
