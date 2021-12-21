#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/matrix_container.h"
#include "../../cpp_graphblas/types.hpp"
#include "../../helpers/cmd_parser/parser_options.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas {
namespace backend {

template<typename T>
class Matrix {
public:
    Matrix() : _format(CSR) {};
    Matrix(Index ncols, Index nrows) : _format(CSR) {};

    ~Matrix() {
        delete data;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        //delete data_socket_dub;
        #endif
        delete transposed_data;
    };

    void build(const VNT *_row_indices,
               const VNT *_col_indices,
               const T *_values,
               const VNT _size,
               const ENT _nnz);

    /* CSR, COO...*/
    LA_Info set_preferred_matrix_format(MatrixStorageFormat format) {
        _format = format;
        return GrB_SUCCESS;
    };

    LA_Info get_format(MatrixStorageFormat* format) const {
        *format = _format;
        return GrB_SUCCESS;
    };

    /* Dense or Sparse*/
    LA_Info setStorage(Storage mat_type) {
        mat_type_ = mat_type;
        return GrB_SUCCESS;
    };

    /* Dense or Sparse*/
    LA_Info get_storage(Storage *mat_type) const {
        *mat_type = mat_type_;
        return GrB_SUCCESS;
    };

    MatrixContainer<T>* get_data() {
        return data;
    }

    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T>* get_data_dub() {
        return data_socket_dub;
    }

    MatrixContainer<T>* get_data_dub() const {
        return data_socket_dub;
    }
    #endif


    const MatrixContainer<T>* get_data() const {
        return data;
    }

    MatrixContainer<T>* get_transposed_data() {
        return transposed_data;
    }

    const MatrixContainer<T>* get_transposed_data() const {
        return transposed_data;
    }

    void get_nrows(VNT* _size) const {
        if (_format == CSR) {
            data->get_size(_size);
        }
    }

    void print() const
    {
        data->print();
    }

    ENT get_nnz() const {return data->get_nnz();};

private:
    MatrixContainer<T> *data;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T> *data_socket_dub;
    #endif

    MatrixContainer<T> *transposed_data;

    MatrixStorageFormat _format;
    Storage mat_type_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(const VNT *_row_indices,
                      const VNT *_col_indices,
                      const T *_values,
                      const VNT _size, // todo remove
                      const ENT _nnz) {
    if (_format == CSR) {
        data = new MatrixCSR<T>;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCSR<T>;
        #endif

        transposed_data = new MatrixCSR<T>;
        cout << "Using CSR matrix format" << endl;
    } else if (_format == LAV) {
        data = new MatrixLAV<T>;
        transposed_data = new MatrixLAV<T>;
        cout << "Using LAV matrix format" << endl;
    } else if (_format == COO) {
        transposed_data = new MatrixCOO<T>;
        data = new MatrixCOO<T>;
        cout << "Using COO matrix format" << endl;
    } else if (_format == CSR_SEG) {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;
        cout << "Using CSR_SEG matrix format" << endl;
    } else if (_format == VECT_GROUP_CSR) {
        data = new MatrixVectGroupCSR<T>;
        transposed_data = new MatrixVectGroupCSR<T>;
        cout << "Using MatrixVectGroupCSR matrix format" << endl;
    } else if (_format == SELL_C) {
        data = new MatrixSellC<T>;
        transposed_data = new MatrixSellC<T>;
        cout << "Using SellC matrix format" << endl;
    } else if(_format == SORTED_CSR) {
        data = new MatrixSortCSR<T>;
        transposed_data = new MatrixSortCSR<T>;
        cout << "Using SortedCSR matrix format" << endl;
    }
    else {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    if(_format == CSR)
        data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
    #endif

    //transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
