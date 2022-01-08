#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "workspace.h"
#include "containers/matrix_container.h"
#include "../../cpp_graphblas/types.hpp"
#include "../../helpers/cmd_parser/parser_options.h"
#include "../la_backend.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Matrix {
public:
    Matrix() : _format(CSR) {};
    Matrix(Index ncols, Index nrows) : _format(CSR) {};

    ~Matrix() {
        delete data;
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        delete data_socket_dub;
        #endif
        delete transposed_data;

        MemoryAPI::free_array(rowdegrees);
        MemoryAPI::free_array(coldegrees);

        delete csr_data;
        delete csc_data;

        delete workspace;
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

    const MatrixCSR<T> *get_csr() const { return csr_data; };
    const MatrixCSR<T> *get_csc() const { return csc_data; };

    const MatrixContainer<T>* get_data() const {
        return data;
    }

    MatrixContainer<T>* get_transposed_data() {
        return transposed_data;
    }

    const MatrixContainer<T>* get_transposed_data() const {
        return transposed_data;
    }

    void get_nrows(VNT* _nrows) const {
        csr_data->get_size(_nrows);
    }

    VNT get_nrows() const {
        VNT nrows;
        csr_data->get_size(&nrows);
        return nrows;
    }

    void get_ncols(VNT* _ncols) const {
        csr_data->get_size(_ncols);
    }

    VNT get_ncols() const {
        VNT ncols;
        csr_data->get_size(&ncols);
        return ncols;
    }

    void print() const
    {
        data->print();
    }

    ENT get_nnz() const {return csr_data->get_nnz();};

    ENT* get_rowdegrees()
    {
        return rowdegrees;
    }

    ENT* get_coldegrees()
    {
        return coldegrees;
    }

    Workspace *get_workspace() const
    {
        return (const_cast <Matrix<T>*> (this))->workspace;
    };
private:
    MatrixContainer<T> *data;
    MatrixContainer<T> *transposed_data;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T> *data_socket_dub;
    #endif

    MatrixCSR<T> *csr_data;
    MatrixCSR<T> *csc_data;

    MatrixStorageFormat _format;
    Storage mat_type_;

    Workspace *workspace;

    ENT *rowdegrees, *coldegrees;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(const VNT *_row_indices,
                      const VNT *_col_indices,
                      const T *_values,
                      const VNT _size, // todo remove
                      const ENT _nnz)
{
    // CSR data creation
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
    csc_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);

    MemoryAPI::allocate_array(&rowdegrees, get_nrows());
    MemoryAPI::allocate_array(&coldegrees, get_ncols());

    #pragma omp parallel for
    for(int i = 0; i < get_nrows(); i++)
    {
        rowdegrees[i] = csr_data->get_degree(i);
    }

    #pragma omp parallel for
    for(int i = 0; i < get_ncols(); i++)
    {
        coldegrees[i] = csc_data->get_degree(i);
    }

    // optimized representation creation
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

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixLAV<T>;
        #endif
        cout << "Using LAV matrix format" << endl;
    } else if (_format == COO) {
        transposed_data = new MatrixCOO<T>;
        data = new MatrixCOO<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCOO<T>;
        #endif
        cout << "Using COO matrix format" << endl;
    } else if (_format == CSR_SEG) {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSegmentedCSR<T>;
        #endif
        cout << "Using CSR_SEG matrix format" << endl;
    } else if (_format == VECT_GROUP_CSR) {
        data = new MatrixVectGroupCSR<T>;
        transposed_data = new MatrixVectGroupCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixVectGroupCSR<T>;
        #endif
        cout << "Using MatrixVectGroupCSR matrix format" << endl;
    } else if (_format == SELL_C) {
        data = new MatrixSellC<T>;
        transposed_data = new MatrixSellC<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSellC<T>;
        #endif
        cout << "Using SellC matrix format" << endl;
    } else if(_format == SORTED_CSR) {
        data = new MatrixSortCSR<T>;
        transposed_data = new MatrixSortCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSortCSR<T>;
        #endif
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

    transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);

    workspace = new Workspace(get_nrows(), get_ncols());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
