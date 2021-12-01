#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class VectGroupCSR;

template <typename T>
struct CSRVertexGroup
{
public:
    CSRVertexGroup();
    ~CSRVertexGroup();

    /* print API */
    void print();

    VNT get_min_nz() const {return min_nz;};
    VNT get_max_nz() const {return max_nz;};

    bool id_in_range(VNT _src_id, VNT _nz_count);

    /* modification API */
    void build(VectGroupCSR<T> *_matrix, VNT _bottom, VNT _top);
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

    template<typename Y>
    friend void SpMV_load_balanced(const VectGroupCSR<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);
    template<typename Y>
    friend void SpMV_vector(const VectGroupCSR<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
