#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define HUB_VERTICES 131072

template <typename T>
class MatrixLAV : public MatrixContainer<T>
{
public:
    MatrixLAV();
    ~MatrixLAV();

    void build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket = 0);
    void print();

    void get_nz(ENT *_nz) const {
        *_nz = nz;
    }

    void get_size(VNT *_size) const {
        *_size = size;
    }

    ENT* get_row() {
        return row_ptr;
    };

    const ENT* get_row() const {
        return row_ptr;
    };

    VNT* get_col() {
        return col_ids;
    };

    const VNT* get_col() const {
        return col_ids;
    };

    T* get_vals() {
        return vals;
    };

    const T* get_vals() const {
        return vals;
    };

    VNT* get_hub() {
        return hub_conversion_array;
    };

    const VNT* get_hub() const {
        return hub_conversion_array;
    };

private:
    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    VNT *hub_conversion_array;

    void alloc(VNT _size, ENT _nz);
    void free();
    void resize(VNT _size, ENT _nz);

    void construct_unsorted_csr(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz);

    bool is_non_zero(VNT _row, VNT _col);
    T get(VNT _row, VNT _col);

    void prepare_hub_data(map<int, int> &_freqs);

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lav_matrix.hpp"
#include "build.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

