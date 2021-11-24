#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixContainer
{
public:
    virtual void build(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, int _socket = 0) = 0;
    virtual void print() = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr/csr_matrix.h"
#include "coo/coo_matrix.h"
#include "seg_csr/seg_csr_matrix.h"
#include "lav/lav_matrix.h"
#include "cell-sigma-C/sigma_matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
