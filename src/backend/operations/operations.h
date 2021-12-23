#ifndef GB_KUN_OPERATIONS_H
#define GB_KUN_OPERATIONS_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "assign_dense.h"
#include "assign_sparse.h"
#include "scatter.h"
#include "reduce.h"
#include "apply_dense.h"
#include "apply_sparse.h"
#include "ewise_mult.h"

namespace lablas{
namespace backend{

template <typename W, typename T, typename M, typename BinaryOpT>
LA_Info assign(Vector<W>*           w,
               const Vector<M>*     mask,
               BinaryOpT            accum,
               const T              val,
               const Index*         indices,
               const Index          nindices,
               Descriptor*          desc)
{
    // Get storage:
    Storage vec_type;
    w->getStorage(&vec_type);
    // 3 cases:
    // 1) SpVec
    // 2) DeVec
    // 3) uninitialized vector
    if (vec_type == GrB_SPARSE) {
        assignSparse(w->getSparse(), mask, accum, val, indices, nindices, desc);
    } else if (vec_type == GrB_DENSE) {
        assignDense(w->getDense(), mask, accum, val, indices, nindices, desc);
    } else {
        //TODO
    }

    return GrB_SUCCESS;
}


template <typename W, typename U, typename M,
        typename BinaryOpT>
    LA_Info assignIndexed(Vector<W>*       w,
                       const Vector<M>* mask,
                       BinaryOpT        accum,
                       const Vector<U>* u,
                       int*             indices,
                       Index            nindices,
                       Descriptor*      desc) {
        Vector<U>* u_t = const_cast<Vector<U>*>(u);

        if (desc->debug()) {
            std::cout << "===Begin assign===\n";
            CHECK(u_t->print());
        }

        Storage u_vec_type;
        CHECK(u_t->getStorage(&u_vec_type));
        if (u_vec_type != GrB_DENSE && u_vec_type != GrB_SPARSE) {
            return GrB_UNINITIALIZED_OBJECT;
        }
        u_t->setStorage(GrB_DENSE);
        w->setStorage(GrB_DENSE);

        scatterIndexed(w->getDense(), mask, accum, u->dense_.d_val_, indices, nindices,
                       desc);

        if (desc->debug()) {
            std::cout << "===End assign===\n";
            CHECK(w->print());
        }
        return GrB_SUCCESS;
    }

template <typename T, typename U,
    typename BinaryOpT, typename MonoidT>
    LA_Info reduce(T*               val,
                   BinaryOpT        accum,
                   MonoidT          op,
                const Vector<U>* u,
                Descriptor*      desc) {
        Vector<U>* u_t = const_cast<Vector<U>*>(u);

        // Get storage:
        Storage vec_type;
        u->getStorage(&vec_type);

        // 2 cases:
        // 1) SpVec
        // 2) DeVec
        if (vec_type == GrB_SPARSE)
            reduceInner(val, accum, op, u->getSparse(), desc);
        else if (vec_type == GrB_DENSE)
            reduceInner(val, accum, op, u->getDense(), desc);
        else
            return GrB_UNINITIALIZED_OBJECT;

        return GrB_SUCCESS;
    }

template <typename W, typename U, typename M,
    typename BinaryOpT,     typename UnaryOpT>
    LA_Info apply(Vector<W>*       w,
               const Vector<M>* mask,
               BinaryOpT        accum,
               UnaryOpT         op,
               const Vector<U>* u,
               Descriptor*      desc) {
        Vector<U>* u_t = const_cast<Vector<U>*>(u);

        Storage u_vec_type;
        u->getStorage(&u_vec_type);

        // sparse variant
        if (u_vec_type == GrB_SPARSE) {
            applySparse(w->getSparse(), mask, accum, op, u_t->getSparse(), desc);
            // dense variant
        } else if (u_vec_type == GrB_DENSE) {
            applyDense(w->getDense(), mask, accum, op, u_t->getDense(), desc);
        } else {
            return GrB_UNINITIALIZED_OBJECT;
        }
        return GrB_SUCCESS;
    }

template <typename W, typename U, typename V, typename M,
    typename BinaryOpT,     typename SemiringT>
    LA_Info eWiseMult(Vector<W>*       w,
                   const Vector<M>* mask,
                   BinaryOpT        accum,
                   SemiringT        op,
                   const Vector<U>* u,
                   const Vector<V>* v,
                   Descriptor*      desc) {
        Vector<U>* u_t = const_cast<Vector<U>*>(u);
        Vector<V>* v_t = const_cast<Vector<V>*>(v);

        Storage u_vec_type;
        Storage v_vec_type;
        u->getStorage(&u_vec_type);
        v->getStorage(&v_vec_type);

        /*
         * \brief 4 cases:
         * 1) sparse x sparse
         * 2) dense  x dense
         * 3) sparse x dense
         * 4) dense  x sparse
         */
        if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_SPARSE) {

            // TODO(ctcyang): Add true sparse-sparse eWiseMult.
            // For now, use dense-sparse.
            /*CHECK(w->setStorage(GrB_SPARSE));
            CHECK(eWiseMultInner(&w->sparse_, mask, accum, op, &u->sparse_,
                &v->sparse_, desc));*/
        }

        if (u_vec_type == GrB_DENSE && v_vec_type == GrB_DENSE) {
            // depending on whether sparse mask is present or not
            if (mask != NULL) {
                Storage mask_type;
                mask->getStorage(&mask_type);
                if (mask_type == GrB_DENSE) {
                    w->setStorage(GrB_DENSE);
                    eWiseMultInner(w->getDense(), mask, accum, op, u->getDense(),
                                         v->getDense(), desc);
                } else if (mask_type == GrB_SPARSE) {
//                    w->setStorage(GrB_SPARSE);
//                    eWiseMultInner(&w->sparse_, &mask->sparse_, accum, op,
//                                         &u->dense_, &v->dense_, desc);
                } else {
                    return GrB_INVALID_OBJECT;
                }
            } else {
                w->setStorage(GrB_DENSE);
                eWiseMultInner(w->getDense(), mask, accum, op, u->getDense(),
                                     v->getDense(), desc);
            }
        } else if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_DENSE) {
            // The boolean here keeps track of whether operators have been reversed.
            // This is important for non-commutative ops i.e. op(a,b) != op(b,a)
//            w->setStorage(GrB_SPARSE);
//            eWiseMultInner(&w->sparse_, mask, accum, op, &u->sparse_,
//                                 &v->dense_, false, desc);
        } else if (u_vec_type == GrB_DENSE && v_vec_type == GrB_SPARSE) {
//            w->setStorage(GrB_SPARSE);
//            eWiseMultInner(&w->sparse_, mask, accum, op, &v->sparse_,
//                                 &u->dense_, true, desc);
        } else {
            return GrB_INVALID_OBJECT;
        }

        return GrB_SUCCESS;
    }
}
}
#endif //GB_KUN_OPERATIONS_H
