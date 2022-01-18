#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixContainer
{
public:
    virtual void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket = 0) = 0;
    virtual void build(vector<vector<VNT>> &_tmp_csr_matrix, int _socket = 0) = 0;

    virtual void print() const = 0;
    virtual void get_size(VNT* _size) const = 0;
    virtual ENT get_nnz() const = 0;
    virtual ENT get_degree(VNT _row) {return 0;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr/csr_matrix.h"
#include "coo/coo_matrix.h"
#include "seg_csr/seg_csr_matrix.h"
#include "lav/lav_matrix.h"
#include "vg_csr/vg_csr.h"
#include "sell_c/sell_c.h"
#include "sort_csr/sort_csr_matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
