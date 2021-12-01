#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

#include "vg/vg.h"
#include "cell_c_vg/cell_c_vg.h"

template <typename T>
class VectGroupCSR : public MatrixContainer<T>
{
public:
    VectGroupCSR();

    ~VectGroupCSR();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();

    VNT get_nz_in_row(VNT _row) {return (row_ptr[_row + 1] - row_ptr[_row]);};
private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    T get(VNT _row, VNT _col);

    int vertex_groups_num;
    CSRVertexGroup<T> *vertex_groups;

    int cell_c_vertex_groups_num;
    int cell_c_start_group;
    CSRVertexGroupCellC<T> *cell_c_vertex_groups;

    void create_vertex_groups();

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz);

    friend class CSRVertexGroup<T>;
    friend class CSRVertexGroupCellC<T>;

    template<typename Y>
    friend void SpMV_load_balanced(const VectGroupCSR<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);
    template<typename Y>
    friend void SpMV_vector(const VectGroupCSR<Y> *_matrix, const DenseVector<Y> *_x, DenseVector<Y> *_y);
};

#include "vg_csr.hpp"
#include "build.hpp"
#include "print.hpp"
#include "vg/vg.hpp"
#include "cell_c_vg/cell_c_vg.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

