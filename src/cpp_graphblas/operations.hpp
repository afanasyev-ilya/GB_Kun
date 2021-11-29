#ifndef GB_KUN_OPERATIONS_HPP
#define GB_KUN_OPERATIONS_HPP
#include "types.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "descriptor.hpp"
#include "../backend/spmv/spmv.h"

namespace lablas {

    /*!
 * Matrix-vector product
 *   w = w + mask .* (A * u)    +: accum
 *                              *: op
 *                             .*: Boolean and
 */

template <typename W, typename M, typename a, typename U,
typename BinaryOpT, typename SemiringT>
LA_Info mxv (Vector<W>*       w,
             const Vector<M>* mask,
             BinaryOpT        accum,
             SemiringT        op,
             const Matrix<a>* A,
             const Vector<U>* u,
             Descriptor*      desc) {

    if (w == NULL || u == NULL || A == NULL || desc == NULL) {
        return GrB_UNINITIALIZED_OBJECT;
    }

    backend::SpMV(A->get_matrix(),  u->get_vector(), w->get_vector(), desc->get_descriptor());
    return GrB_SUCCESS;
}

}




#endif //GB_KUN_OPERATIONS_HPP
