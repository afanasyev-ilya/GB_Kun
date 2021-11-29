#ifndef GB_KUN_MATRIX_HPP
#define GB_KUN_MATRIX_HPP

#include "../backend/matrix/matrix.h"
#include "../helpers/utils.hpp"
#include "types.hpp"
#include <vector>
#include "../backend/la_backend.h"

namespace lablas {

template<typename T>
class Matrix {
public:
    Matrix() : _matrix() {};
    Matrix(Index nrows, Index ncols) : _matrix(nrows, ncols) {}

    backend::Matrix<T>* get_matrix() {
        return &_matrix;
    }

    const backend::Matrix<T>* get_matrix() const {
        return &_matrix;
    }

    LA_Info get_storage(Storage* _mat_type) const {
        return _matrix.get_storage(_mat_type);
    }

    LA_Info set_preferred_matrix_format(const MatrixStorageFormat _format) {
        return _matrix.set_preferred_matrix_format(_format);
    }

    template <typename BinaryOpT>
    LA_Info build (const std::vector<VNT>*   row_indices,
                   const std::vector<VNT>*   col_indices,
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
        /* doubling nvlas because _nz = nvals in implementation - TODO remove*/
        if (!row_indices->empty()) {
            _matrix.build(row_indices->data(), col_indices->data(), values->data(), nvals, row_indices->size());
        }
        return GrB_SUCCESS;
    }

    void print()
    {
        _matrix.print();
    }

private:
    backend::Matrix<T> _matrix;
};

}

#endif //GB_KUN_MATRIX_HPP
