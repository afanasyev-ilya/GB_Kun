#ifndef GB_KUN_MATRIX_HPP
#define GB_KUN_MATRIX_HPP

/// @file matrix.hpp
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Matrix class frontend wrapper for cpp usage
/// @details Describes GraphBLAS methods for matrix object, invokes functions and methods of backend matrix object
/// @date June 12, 2022

#include "../backend/matrix/matrix.h"
#include "types.hpp"
#include <vector>
#include "../backend/la_backend.h"

namespace lablas {

template<typename T>
class Matrix {
public:
    /**
     * Create a new Matrix object with default sizes
     * @brief Default constructor
     * @see Matrix(Index nrows, Index ncols)
     */
    Matrix() : _matrix() {};

    /**
     * Create a new Matrix object with particular sizes
     * @brief Parameter constructor
     * @param nrows The number of rows in new matrix
     * @param ncols The number of columns in new matrix
     * @see Matrix()
     */
    Matrix(Index nrows, Index ncols) : _matrix(nrows, ncols) {}

    /**
     * @brief Print graphviz image of matrix object
     * @param[in] file_name The name of output file
     * @param[in] label_vector Labels are to be put into vertices of a visualized graph
     * @return Nothing
     */
    template<class U>
    void print_graphviz(string file_name, backend::Vector<U>* label_vector) {
        _matrix.print_graphviz(file_name, backend::VisualizationMode::VISUALISE_AS_UNDIRECTED, label_vector);
    }

    /**
     * @brief Get arbitrary pointer to the matrix bakcend member of matrix class
     * @return matrix Pointer to the backend matrix
     */
    backend::Matrix<T>* get_matrix() {
        return &_matrix;
    }

    /**
     * @brief Get constant pointer to the matrix bakcend member of matrix class
     * @return matrix Const pointer to the backend matrix
     */
    const backend::Matrix<T>* get_matrix() const {
        return &_matrix;
    }

    /**
     * @brief Get the type of main backend container (sparse or dense representation)
     * @param[out] _mat_type Type of current matrix storage
     * @return la_info Flag for the correctness check
     */
    LA_Info get_storage(Storage* _mat_type) const {
        return _matrix.get_storage(_mat_type);
    }

    /**
     * @brief Set the sparse matrix storage format (CSR, COO, etc.)
     * @param[in] _mat_type Type of desired storage format
     * @return la_info Flag for the correctness check
     */
    LA_Info set_preferred_matrix_format(const MatrixStorageFormat _format) {
        return _matrix.set_preferred_matrix_format(_format);
    }

    /**
     * @brief Get the number of rows allocated for matrix (vertices in a represented graph)
     * @param[out] _nrows Number of rows in dense matrix (maximum number of rows in sparse matrix)
     * @return la_info Flag for the correctness check
     * @see nrows()
     */
    LA_Info get_nrows(Index* _nrows) const {
        *_nrows = _matrix.get_nrows();
        return GrB_SUCCESS;
    }

    /**
    * @brief Get the number of rows allocated for matrix (vertices in a represented graph)
    * @return nrows Number of rows in dense matrix (maximum number of rows in sparse matrix)
    */
    Index nrows() const{
        return _matrix.get_nrows();
    }

    /**
    * @brief Get the number of columns allocated for matrix (vertices in a represented graph)
    * @return ncols Number of columns in dense matrix (maximum number of rows in sparse matrix)
    */
    Index ncols() const{
        return _matrix.get_ncols();
    }

    /**
    * @brief Get the number of columns allocated for matrix (vertices in a represented graph)
    * @param[out] _ncols Number of columns in dense matrix (maximum number of colums in sparse matrix)
    * @return la_info Flag for the correctness check
    */
    LA_Info get_ncols(Index* _ncols) const{
        *_ncols = _matrix.get_ncols();
        return GrB_SUCCESS;
    }

    /**
    * @brief Get the total number or non-NULL elements in a matrix
    * @param[out] _nvals Total number or non-NULL elements in a matrix
    * @return la_info Flag for the correctness check
    */
    LA_Info get_nvals(Index* _nvals) const{
        *_nvals = _matrix.get_nnz();
        return GrB_SUCCESS;
    }

    /**
    * @brief Get the total number or non-NULL elements in a matrix
    * @param[out] _nvals Total number or non-NULL elements in a matrix
    * @return la_info Flag for the correctness check
    */
    Index* get_rowdegrees() { return _matrix.get_rowdegrees(); }
    [[nodiscard]] const Index* get_rowdegrees() const { return _matrix.get_rowdegrees(); }

    Index* get_coldegrees() { return _matrix.get_coldegrees(); }
    [[nodiscard]] const Index* get_coldegrees() const { return _matrix.get_coldegrees(); }

    LA_Info transpose() { return _matrix.transpose(); }

    template <typename BinaryOpT>
    LA_Info build (const std::vector<Index>*   row_indices,
                   const std::vector<Index>*   col_indices,
                   const std::vector<T>*     values,
                   Index                     nvals,
                   BinaryOpT                 dup,
                   char*                     dat_name = nullptr)
    {
        if (row_indices == nullptr || col_indices == nullptr || values == nullptr) {
            return GrB_NULL_POINTER;
        }
        if (row_indices->empty() && col_indices->empty() && values->empty()) {
            return GrB_NO_VALUE;
        }
        if (row_indices->size() != col_indices ->size()) {
            return GrB_DIMENSION_MISMATCH;
        }

        if (!row_indices->empty()) {
            _matrix.build(row_indices->data(), col_indices->data(), values->data(), row_indices->size());
        }
        return GrB_SUCCESS;
    }

    LA_Info build_from_csr_arrays(const Index* _row_ptrs,
                                  const Index *_col_ids,
                                  const T *_values,
                                  Index _nrows,
                                  Index _nvals)
    {
        if (_row_ptrs == nullptr || _col_ids == nullptr || _values == nullptr) {
            return GrB_NULL_POINTER;
        }

        _matrix.build_from_csr_arrays(_row_ptrs, _col_ids, _values, _nrows, _nvals);
        return GrB_SUCCESS;
    }

    void init_from_mtx(const string &_mtx_name)
    {
        _matrix.init_from_mtx(_mtx_name);
    }

    void print() const
    {
        _matrix.print();
    }

    Index get_nnz() const {return _matrix.get_nnz();};

    void sort_csr_columns(const string& mode)
    {
        _matrix.sort_csr_columns(mode);
    }

    void sort_csc_rows(const string& mode)
    {
        _matrix.sort_csc_rows(mode);
    }

    bool is_symmetric() const {
        return _matrix.is_symmetric();
    }

    void to_symmetric() { _matrix.to_symmetric(); };

    friend bool operator==(const Matrix<T>& _lhs, const Matrix<T>& _rhs) {
        return _lhs._matrix == _rhs._matrix;
    }

    friend bool operator!=(const Matrix<T>& _lhs, const Matrix<T>& _rhs) {
        return !(_lhs._matrix == _rhs._matrix);
    }
private:
    backend::Matrix<T> _matrix;
};

}

#endif //GB_KUN_MATRIX_HPP
