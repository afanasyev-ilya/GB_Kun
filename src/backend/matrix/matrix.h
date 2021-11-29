#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/matrix_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Matrix
{
public:
    Matrix() {format = CSR;};
    ~Matrix()
    {
        delete data;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        delete data_socket_dub;
        #endif
        delete transposed_data;
    };

    void build(VNT *_row_indices,
               VNT *_col_indices,
               T *_values,
               const VNT _size, // todo remove
               const ENT _nz);

    void set_preferred_format(MatrixStorageFormat _format) {format = _format;};
private:
    MatrixContainer<T> *data;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T> *data_socket_dub;
    #endif

    MatrixContainer<T> *transposed_data;

    MatrixStorageFormat format;

    template<typename Y>
    friend void SpMV(Matrix<Y> &_matrix,
                     Vector<Y> &_x,
                     Vector<Y> &_y,
                     Descriptor &_desc);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(VNT *_row_indices,
                      VNT *_col_indices,
                      T *_values,
                      const VNT _size, // todo remove
                      const ENT _nz)
{
    if(format == CSR)
    {
        data = new MatrixCSR<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCSR<T>;
        #endif

        transposed_data = new MatrixCSR<T>;
        cout << "Using CSR matrix" << endl;
    }
    else if(format == LAV)
    {
        data = new MatrixLAV<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixLAV<T>;
        #endif

        transposed_data = new MatrixLAV<T>;
        cout << "Using LAV matrix" << endl;
    }
    else if(format == COO)
    {
        data = new MatrixCOO<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCOO<T>;
        #endif

        transposed_data = new MatrixCOO<T>;
        cout << "Using COO matrix" << endl;
    }
    else if(format == CSR_SEG)
    {
        data = new MatrixSegmentedCSR<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSegmentedCSR<T>;
        #endif

        transposed_data = new MatrixSegmentedCSR<T>;
        cout << "Using CSR_SEG matrix" << endl;
    }
    else if(format == CELL_SIGMA_C)
    {
        data = new MatrixCellSigmaC<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCellSigmaC<T>;
        #endif

        transposed_data = new MatrixCellSigmaC<T>;
        cout << "Using CELL_SIGMA_C matrix" << endl;
    }
    else
    {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    data->build(_row_indices, _col_indices, _values, _size, _nz, 0);
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nz, 1);
    #endif

    transposed_data->build(_col_indices, _row_indices, _values, _size, _nz, 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
