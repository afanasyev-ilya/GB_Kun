#pragma once

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

    VNT get_size() {return size;};
    ENT get_non_zeroes_num() {return non_zeroes_num;};

    VNT *get_row_ptr(){return row_ptr;};
    T *get_vals(){return vals;};
    ENT *get_col_ids(){return col_ids;};

    void resize(VNT _size, ENT _non_zeroes_num);

    void import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _non_zeroes_num);
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "csr_matrix.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

