#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "workspace.h"
#include "containers/matrix_container.h"
#include "../../cpp_graphblas/types.hpp"
#include "../../helpers/cmd_parser/parser_options.h"
#include "../la_backend.h"
#include <atomic>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <class T>
inline T my_fetch_add(T *ptr, T val) {
#ifdef _OPENMP //201511
        T t;
#pragma omp atomic capture
    { t = *ptr; *ptr += val; }
    return t;
#endif
}


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

    template<class U>
    void print_graphviz(string _file_name, VisualizationMode _visualisation_mode, Vector<U>* label_vector) {

        if (csr_data->get_num_rows() == label_vector->get_size()) {
            std::cout << "Label vector and matrix match each other" << std::endl;
        } else {
            std::cout << "Error in dims mismatch" << std::endl;
        }

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
            dot_output << cur_vertex << " [label = \" " << label_vector->getDense()->get_vals()[cur_vertex] << " \"];" << endl;
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

    void get_nrows(VNT* _nrows) const {
        *_nrows = csr_data->get_num_rows();
    }

    VNT get_nrows() const {
        return csr_data->get_num_rows();
    }

    void get_ncols(VNT* _ncols) const {
        *_ncols = csr_data->get_ncols();
    }

    VNT get_ncols() const {
        return csr_data->get_num_cols();
    }

    void print() const { csr_data->print(); }

    ENT get_nnz() const {return csr_data->get_nnz();};

    ENT* get_rowdegrees() { return csr_data->get_rowdegrees(); }

    ENT* get_coldegrees() { return csc_data->get_rowdegrees(); }

    Workspace *get_workspace() const { return (const_cast <Matrix<T>*> (this))->workspace; };

    void sort_csr_columns(const string& mode);

    void sort_csc_rows(const string& mode);

    void transpose(void);

    void transpose_parallel(void) {
        memset(csc_data->get_row_ptr(),0, (csc_data->get_num_rows() + 1) * sizeof(Index));
        memset(csc_data->get_col_ids(),0, csc_data->get_nnz()* sizeof(Index));
        memset(csc_data->get_vals(),0, csc_data->get_nnz()* sizeof(T));

        VNT csr_ncols = csr_data->get_num_cols();
        VNT csr_nrows = csr_data->get_num_rows();
        auto dloc = new int[csr_data->get_nnz()];

        Index temp;
        Index* row_ptr = csc_data->get_row_ptr();

#pragma omp parallel for schedule(dynamic) shared(csr_nrows, csr_ncols, row_ptr, dloc)
        for (int i = 0; i < csr_nrows; i++) {
            for (int j = csr_data->get_row_ptr()[i]; j < csr_data->get_row_ptr()[i + 1]; j++) {
                dloc[j] = my_fetch_add(&row_ptr[csr_data->get_col_ids()[j] + 1], static_cast<Index>(1));
            }
        }
        /*TODO parallel scan*/
        for (Index i = 1; i < csr_ncols + 1; i++){
            csc_data->get_row_ptr()[i] += csc_data->get_row_ptr()[i - 1];
        }
#pragma omp parallel for schedule(dynamic) shared(csr_nrows, csr_ncols, row_ptr, dloc)
        for (Index i = 0; i < csr_nrows; i++) {
            for (Index j =csr_data->get_row_ptr()[i]; j < csr_data->get_row_ptr()[i + 1]; j++) {
                auto loc = csc_data->get_row_ptr()[csr_data->get_col_ids()[j]] + dloc[j];
                csc_data->get_col_ids()[loc] = i;
                csc_data->get_vals()[loc] = csr_data->get_vals()[j];
            }
        }

    }
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
