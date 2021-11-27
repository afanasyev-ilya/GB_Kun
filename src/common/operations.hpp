#ifndef GB_KUN_OPERATIONS_HPP
#define GB_KUN_OPERATIONS_HPP
#include "types.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "descriptor.hpp"

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

                backend::Descriptor& _desc= desc->get_descriptor();
                backend::Matrix<a>& _matrix = A->get_matrix();
            }
}




#endif //GB_KUN_OPERATIONS_HPP
