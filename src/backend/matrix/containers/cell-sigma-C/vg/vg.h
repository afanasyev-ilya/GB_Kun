#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
public:
    CSRVertexGroup();
    ~CSRVertexGroup();

    /* prVNT API */
    void prVNT();

    /* get API */
    VNT *get_ids() {return ids;};
    VNT get_size() const {return size;};
    VNT get_max_size() const {return max_size;};
    ENT get_total_nz() const {return total_nz;};

    VNT get_min_nz() const {return min_nz;};
    VNT get_max_nz() const {return max_nz;};

    bool id_in_range(VNT _src_id, VNT _nz_count);

    /* modification API */
    template <typename T>
    void build(MatrixCellSigmaC<T> *_matrix, VNT _bottom, VNT _top);
    void copy(CSRVertexGroup &_other_group);
    void add_vertex(VNT _src_id);
    void clear() {size = 0;};

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroup &_full_group, CopyCond copy_cond, VNT *_buffer);

    void resize(VNT _new_size);
private:
    VNT *ids;
    VNT size;
    VNT max_size;
    ENT total_nz;

    VNT min_nz, max_nz;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
