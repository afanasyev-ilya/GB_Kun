#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixCOO
{
private:
    VNT size;
    ENT nz;

    VNT *row_ids;
    VNT *col_ids;
    T *vals;

    void alloc(VNT _size, ENT _nz);
    void free();
public:
    MatrixCOO();
    ~MatrixCOO();

    VNT get_size() {return size;};
    ENT get_nz() {return nz;};

    ENT *get_row_ids(){return row_ids;};
    ENT *get_col_ids(){return col_ids;};
    T *get_vals(){return vals;};

    void resize(VNT _size, ENT _nz);

    void import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz);
    void print();

    void import(int *_row_ids, int *_col_ids, T *_vals, int _size, int _nz, bool _optimized);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "coo_matrix.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

