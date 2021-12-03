#ifndef GB_KUN_OPERATIONS_HPP
#define GB_KUN_OPERATIONS_HPP
#include "types.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "descriptor.hpp"
#include "dimensions.hpp"
#include "../backend/spmv/spmv.h"
#include "../backend/operations/operations.h"

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

        backend::SpMV(A->get_matrix(), u->get_vector(), w->get_vector(), NULL,  op);
        return GrB_SUCCESS;
    }


    /*!
     * Assign constant to vector subset
     *   w[indices] = w[indices] + val   +: accum
     *                                  .*: Boolean and
     */
template <typename W, typename M, typename T, typename I,
    typename BinaryOpT>
    LA_Info assign(Vector<W>*       w,
                Vector<M>*       mask,
                BinaryOpT        accum,
                T                val,
                const Vector<I>* indices,
                Index            nindices,
                Descriptor*      desc) {
        // Null pointer check
        if (w == NULL || desc == NULL)
            return GrB_UNINITIALIZED_OBJECT;

        // Dimension check
        // -only have one case (no transpose option)
//        checkDimSizeSize(w, mask, "w.size  != mask.size");

        auto                 mask_t = (mask == NULL) ? NULL : mask->get_vector();
        backend::Descriptor* desc_t = (desc == NULL) ? NULL : desc->get_descriptor();
        auto              indices_t = (indices == NULL) ? NULL : indices->get_vector();

        return backend::assign(w->get_vector(), mask_t, accum, val, indices_t, nindices,
                               desc_t);
    }

template <typename W, typename M, typename U,
    typename BinaryOpT>
    LA_Info assignIndexed(Vector<W>*       w,
                       const Vector<M>* mask,
                       BinaryOpT        accum,
                       const Vector<U>* u,
                       int*             indices,
                       Index            nindices,
                       Descriptor*      desc) {
        // Null pointer check
        if (w == NULL || u == NULL || desc == NULL)
            return GrB_UNINITIALIZED_OBJECT;

        // Dimension check
        // -only have one case (no transpose option)
//        checkDimSizeSize(w, mask, "w.size  != mask.size");

        auto                 mask_t = (mask == NULL) ? NULL : mask->get_vector();
        backend::Descriptor* desc_t = (desc == NULL) ? NULL : desc->get_descriptor();

        return backend::assignIndexed(w->get_vector(), mask_t, accum, u->get_vector(),
                                      indices, nindices, desc_t);
    }

    /*!
 * Reduction of vector to form scalar
 *   val = val + \sum_i u(i) for all i    +: accum
 *                                      sum: op
 */
    template <typename T, typename U,
    typename BinaryOpT, typename MonoidT>
    LA_Info reduce(T*               val,
                BinaryOpT        accum,
                MonoidT          op,
                const Vector<U>* u,
                Descriptor*      desc) {
        if (val == NULL || u == NULL)
            return GrB_UNINITIALIZED_OBJECT;

        backend::Descriptor* desc_t = (desc == NULL) ? NULL : desc->get_descriptor();

        return backend::reduce(val, accum, op, u->get_vector(), desc_t);
    }

    /*!
 * Apply unary operation to vector
 *   w = w + mask .* op(u)    +: accum
 *                           .*: Boolean and
 */
template <typename W, typename M, typename U,
        typename BinaryOpT,     typename UnaryOpT>
        LA_Info apply(Vector<W>*       w,
                   const Vector<M>* mask,
                   BinaryOpT        accum,
                   UnaryOpT         op,
                   const Vector<U>* u,
                   Descriptor*      desc) {
            // Null pointer check
            if (w == NULL || u == NULL)
                return GrB_UNINITIALIZED_OBJECT;

            const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : mask->get_vector();
            backend::Descriptor*      desc_t = (desc == NULL) ? NULL : desc->get_descriptor();

            return backend::apply(w->get_vector(), mask_t, accum, op, u->get_vector(), desc_t);
        }


/*!
 * Element-wise multiply of two vectors
 *   w = w + mask .* (u .* v)    +: accum
 *                 2     1    1).*: op (semiring multiply)
 *                            2).*: Boolean and
 */
template <typename W, typename M, typename U, typename V,
    typename BinaryOpT,     typename SemiringT>
    LA_Info eWiseMult(Vector<W>*       w,
                   const Vector<M>* mask,
                   BinaryOpT        accum,
                   SemiringT        op,
                   const Vector<U>* u,
                   const Vector<V>* v,
                   Descriptor*      desc) {
        // Null pointer check
        if (w == NULL || u == NULL || v == NULL || desc == NULL)
            return GrB_UNINITIALIZED_OBJECT;


        const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : mask->get_vector();
        backend::Descriptor*      desc_t = (desc == NULL) ? NULL : desc->get_descriptor();

        return backend::eWiseMult(w->get_vector(), mask_t, accum, op, u->get_vector(),
                                  v->get_vector(), desc_t);
    }

}

#endif //GB_KUN_OPERATIONS_HPP
