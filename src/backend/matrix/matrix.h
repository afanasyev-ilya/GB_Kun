#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "containers/matrix_container.h"
#include "../../common/types.hpp"
#include "../helpers/cmd_parser/parser_options.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas {
namespace backend {

template<typename T>
class Matrix {
public:
    Matrix() : format(CSR) {};
    Matrix(Index ncols, Index nrows) : format(CSR) {};

    ~Matrix() {
        delete data;
#ifdef __USE_SOCKET_OPTIMIZATIONS__
        delete data_socket_dub;
#endif
        delete transposed_data;
    };

    void build(const VNT *_row_indices,
               const VNT *_col_indices,
               const T *_values,
               const VNT _size, // todo remove
               const ENT _nz);

    LA_Info set_preferred_format(MatrixStorageFormat _format) {
        format = _format;
        return GrB_SUCCESS;
    };

    LA_Info getStorage(Storage *mat_type) {
        *mat_type = mat_type_;
        return GrB_SUCCESS;
    };

private:
    MatrixContainer<T> *data;
#ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T> *data_socket_dub;
#endif

    MatrixContainer<T> *transposed_data;


    /* TODO Change to ENV variable! Now only manual change */
    MatrixStorageFormat format;
    Storage mat_type_;

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
                      const ENT _nz) {
    if (format == CSR) {
        data = new MatrixCSR<T>;
#ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCSR<T>;
#endif

        transposed_data = new MatrixCSR<T>;
    } else if (format == LAV) {
        data = new MatrixLAV<T>;
        transposed_data = new MatrixLAV<T>;
    } else if (format == COO) {
        data = new MatrixCOO<T>;
        transposed_data = new MatrixCOO<T>;
    } else if (format == CSR_SEG) {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;
    } else {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    data->build(_row_indices, _col_indices, _values, _size, _nz, 0);
#ifdef __USE_SOCKET_OPTIMIZATIONS__
    data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nz, 1);
#endif

    transposed_data->build(_col_indices, _row_indices, _values, _size, _nz, 0);
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
