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


    enum VisualizationMode {
        VISUALISE_AS_DIRECTED,
        VISUALISE_AS_UNDIRECTED
    };
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Matrix {
public:
    Matrix();

    Matrix(Index ncols, Index nrows);

    ~Matrix();

    void build(const VNT *_row_indices,
               const VNT *_col_indices,
               const T *_values,
               const VNT _size,
               const ENT _nnz);

    void init_from_mtx(const string &_mtx_file_name);

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
    LA_Info get_storage(Storage *mat_type) const {
        *mat_type = GrB_SPARSE;
        return GrB_SUCCESS;
    };

    void print_graphviz(string _file_name, VisualizationMode _visualisation_mode) {

        ofstream dot_output(_file_name.c_str());

        string edge_symbol;
        if(_visualisation_mode == VISUALISE_AS_DIRECTED)
        {
            dot_output << "digraph G {" << endl;
            edge_symbol = " -> ";
        }
        else if(_visualisation_mode == VISUALISE_AS_UNDIRECTED)
        {
            dot_output << "graph G {" << endl;
            edge_symbol = " -- ";
        }
        auto num_vertices = get_nrows();
        for(int cur_vertex = 0; cur_vertex < num_vertices; cur_vertex++)
        {
            dot_output << cur_vertex << " [label = \" " << csr_data->get_vals()[cur_vertex] << " \"];" << endl;
        }

        for(int cur_vertex = 0; cur_vertex < num_vertices; cur_vertex++)
        {
            int src_id = cur_vertex;
            for(long long edge_pos = csr_data->get_row_ptr()[cur_vertex]; edge_pos < csr_data->get_row_ptr()[cur_vertex + 1]; edge_pos++)
            {
                int dst_id = csr_data->get_col_ids()[edge_pos];
                dot_output << src_id << edge_symbol << dst_id /*<< " [label = \" " << weight << " \"];"*/ << endl;
            }
        }

        dot_output << "}";
        dot_output.close();
    }

    MatrixContainer<T>* get_data() {
        return data;
    }

    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T>* get_data_dub() { return data_socket_dub; }

    MatrixContainer<T>* get_data_dub() const { return data_socket_dub; }
    #endif

    const MatrixCSR<T> *get_csr() const { return csr_data; };
    const MatrixCSR<T> *get_csc() const { return csc_data; };

    const MatrixContainer<T>* get_data() const { return data; }

    MatrixContainer<T>* get_transposed_data() { return transposed_data; }

    const MatrixContainer<T>* get_transposed_data() const { return transposed_data; }

    void get_nrows(VNT* _nrows) const { csr_data->get_size(_nrows); }

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

    void print() const { csr_data->print(); }

    ENT get_nnz() const {return csr_data->get_nnz();};

    ENT* get_rowdegrees() { return rowdegrees; }

    ENT* get_coldegrees() { return coldegrees; }

    Workspace *get_workspace() const { return (const_cast <Matrix<T>*> (this))->workspace; };
private:
    MatrixContainer<T> *data;
    MatrixContainer<T> *transposed_data;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    MatrixContainer<T> *data_socket_dub;
    #endif

    MatrixCSR<T> *csr_data;
    MatrixCSR<T> *csc_data;

    MatrixStorageFormat _format;

    Workspace *workspace;

    ENT *rowdegrees, *coldegrees;

    void read_mtx_file_pipelined(const string &_mtx_file_name,
                                 vector<vector<pair<VNT, T>>> &_csr_matrix,
                                 vector<vector<pair<VNT, T>>> &_csc_matrix);

    void init_optimized_structures();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "matrix.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
