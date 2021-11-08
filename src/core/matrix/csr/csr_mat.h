#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCSR
{
private:
    VNT size;
    ENT non_zeroes_num;

    VNT *row_ptr;
    T *vals;
    ENT *col_ids;

    void alloc(VNT _size, ENT _non_zeroes_num);
    void free();

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, T *_vals, VNT _size, ENT _non_zeroes_num);

    bool is_non_zero(int _row, int _col);
    T get(int _row, int _col);
public:
    MatrixCSR();
    ~MatrixCSR();

    void resize(VNT _size, ENT _non_zeroes_num);

    void import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _non_zeroes_num);
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_mat.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

