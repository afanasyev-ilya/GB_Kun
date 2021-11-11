#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define HUB_VERTICES 15 //131072

template <typename T>
class MatrixLAV
{
private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    VNT *hub_conversion_array;

    void alloc(VNT _size, ENT _nz);
    void free();

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(int _row, int _col);
    T get(int _row, int _col);

    void prepare_hub_data(map<int, int> &_freqs);
public:
    MatrixLAV();
    ~MatrixLAV();

    VNT get_size() {return size;};
    ENT get_nz() {return nz;};

    ENT *get_row_ptr(){return row_ptr;};
    T *get_vals(){return vals;};
    VNT *get_col_ids(){return col_ids;};
    VNT *get_hub_conversion_array(){return hub_conversion_array;};

    void resize(VNT _size, ENT _nz);

    void import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz);
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lav_matrix.hpp"
#include "import.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

