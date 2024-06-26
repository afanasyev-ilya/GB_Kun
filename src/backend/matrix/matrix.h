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
               ENT _nnz);

    void build_from_csr_arrays(const ENT* _row_ptrs,
                               const VNT *_col_ids,
                               const T *_values,
                               Index _nrows,
                               Index _nnz);

    void build(vector<vector<pair<VNT, T>>>& csc_tmp_matrix,
               vector<vector<pair<VNT, T>>>& csr_tmp_matrix);

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

    const MatrixCSR<T> *get_csr() const { return csr_data; };
    const MatrixCSR<T> *get_csc() const { return csc_data; };

    const MatrixContainer<T>* get_data() const { return data; }

    MatrixContainer<T>* get_transposed_data() { return transposed_data; }

    const MatrixContainer<T>* get_transposed_data() const { return transposed_data; }

    void get_nrows(VNT* _nrows) const {
        *_nrows = csr_data->get_num_rows();
    }

    [[nodiscard]] VNT get_nrows() const {
        return csr_data->get_num_rows();
    }

    void get_ncols(VNT* _ncols) const {
        *_ncols = csr_data->get_ncols();
    }

    [[nodiscard]] VNT get_ncols() const {
        return csr_data->get_num_cols();
    }

    void print() const { csr_data->print(); }

    [[nodiscard]] ENT get_nnz() const {return csr_data->get_nnz();};

    [[nodiscard]] ENT* get_rowdegrees() { return csr_data->get_rowdegrees(); }
    [[nodiscard]] const ENT* get_rowdegrees() const { return csr_data->get_rowdegrees(); }

    [[nodiscard]] ENT* get_coldegrees() { return csc_data->get_rowdegrees(); }
    [[nodiscard]] const ENT* get_coldegrees() const { return csc_data->get_rowdegrees(); }

    [[nodiscard]] Workspace *get_workspace() const { return (const_cast <Matrix<T>*> (this))->workspace; };

    void sort_csr_columns(const string& mode);

    void sort_csc_rows(const string& mode);

    LA_Info transpose();
    LA_Info csr_to_csc();

    void transpose_sequential(void);
    void transpose_parallel(void);

    [[nodiscard]] bool is_symmetric() const {
        //return (*csr_data == *csc_data);
        return csr_data->is_symmetric();
    }

    void to_symmetric() { csr_data->to_symmetric(); csc_data->deep_copy(csr_data); };

    friend bool operator==(const Matrix<T>& _lhs, const Matrix<T>& _rhs) {
        return (*(_lhs.csr_data) == *(_rhs.csr_data)) && (*(_lhs.csc_data) == *(_rhs.csc_data));
    }
private:
    MatrixContainer<T> *data;
    MatrixContainer<T> *transposed_data;

    MatrixCSR<T> *csr_data;
    MatrixCSR<T> *csc_data;

    MatrixStorageFormat _format;

    Workspace *workspace;

    void read_mtx_file_pipelined(const string &_mtx_file_name,
                                 vector<vector<pair<VNT, T>>> &_csr_matrix,
                                 vector<vector<pair<VNT, T>>> &_csc_matrix);

    void binary_read_mtx_file_pipelined(const string &_mtx_file_name,
                                        vector<vector<pair<VNT, T>>> &_csr_matrix,
                                        vector<vector<pair<VNT, T>>> &_csc_matrix);

    void binary_read_mtx_file(const string &_mtx_file_name,
                              vector<vector<pair<VNT, T>>> &_csr_matrix,
                              vector<vector<pair<VNT, T>>> &_csc_matrix);

    void read_mtx_file_sequential(const string &_mtx_file_name,
                                  vector<vector<pair<VNT, T>>> &_csr_matrix,
                                  vector<vector<pair<VNT, T>>> &_csc_matrix);

    void init_optimized_structures();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "matrix.hpp"
#include "transpose.hpp"
#include "build.hpp"
#include "read_file.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
