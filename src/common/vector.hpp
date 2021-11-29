#ifndef GB_KUN_VECTOR_HPP
#define GB_KUN_VECTOR_HPP

#include "../backend/vector/vector.h"
#include "../helpers/utils.hpp"
#include "types.hpp"
#include <vector>

namespace lablas {

template<typename T>
class Vector {
public:
    Vector() : _vector() {}
    explicit Vector(Index nsize) : _vector(nsize) {}

    ~Vector() {}


    //    template <typename BinaryOpT>
    //    LA_Info build (const std::vector<VNT>*   row_indices,
    //                   const std::vector<VNT>*   col_indices,
    //                   const std::vector<T>*     values,
    //                   Index                     nvals,
    //                   BinaryOpT                 dup,
    //                   char*                     dat_name = nullptr) {
    //        if (row_indices == nullptr || col_indices == nullptr || values == nullptr) {
    //            return GrB_NULL_POINTER;
    //        }
    //        if (row_indices->empty() && col_indices->empty() && values->empty()) {
    //            return GrB_NO_VALUE;
    //        }
    //        if (row_indices->size() != col_indices ->size()) {
    //            return GrB_DIMENSION_MISMATCH;
    //        }
    //        /* doubling nvlas because _nz = nvals in implementation - TODO remove*/
    //        if (!row_indices->empty()) {
    //            _matrix.build(row_indices->data(), col_indices->data(), values->data(), nvals, nvals);
    //        }
    //        return GrB_SUCCESS;
    //    }

    LA_Info fill(T val) {
        _vector.set_constant(val);
        return GrB_SUCCESS;
    }

    backend::Vector<T>* get_vector() {
        return &_vector;
    }

    const backend::Vector<T>* get_vector() const {
        return &_vector;
    }


    //    template <typename BinaryOpT>
    //    Info build(const std::vector<Index>* indices,
    //               const std::vector<T>*     values,
    //               Index                     nvals,
    //               BinaryOpT                 dup);
    //    Info build(const std::vector<T>* values,
    //               Index                 nvals);
    //    Info build(Index* indices,
    //               T*     values,
    //               Index  nvals);
    //    Info build(T*    values,
    //               Index nvals);


private:
    backend::Vector<T> _vector;
};
}
#endif //GB_KUN_VECTOR_HPP
