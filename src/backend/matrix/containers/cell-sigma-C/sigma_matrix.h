#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vg/vg.h"
#include "cell_c_vg/cell_c_vg.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CSR_VERTEX_GROUPS_NUM 6

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCellSigmaC : public MatrixContainer<T>
{
public:
    MatrixCellSigmaC();

    ~MatrixCellSigmaC();

    void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, VNT _socket = 0);
    void prVNT();
private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    CSRVertexGroup vertex_groups[CSR_VERTEX_GROUPS_NUM];

    VNT cell_c_vertex_groups_num;
    //CSRVertexGroupCellC cell_c_vertex_groups[CSR_VERTEX_GROUPS_NUM];

    void create_vertex_groups();

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    void construct_unsorted_csr(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz);

    friend class CSRVertexGroup;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vg/vg.hpp"
#include "cell_c_vg/cell_c_vg.hpp"
#include "sigma_matrix.hpp"
#include "build.hpp"
#include "print.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

