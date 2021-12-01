#ifndef GB_KUN_OPERATIONS_H
#define GB_KUN_OPERATIONS_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "assign_dense.h"
#include "assign_sparse.h"
#include "scatter.h"

namespace lablas{
namespace backend{

template <typename W, typename T, typename M,
    typename BinaryOpT>
    LA_Info assign(Vector<W>*           w,
                Vector<M>*           mask,
                BinaryOpT            accum,
                T                    val,
                const Vector<Index>* indices,
                Index                nindices,
                Descriptor*          desc) {
        if (desc->debug()) {
            std::cout << "===Begin assign===\n";
            std::cout << "Input: " << val << std::endl;
        }

        // Get storage:
        Storage vec_type;
        CHECK(w->getStorage(&vec_type));

        // 3 cases:
        // 1) SpVec
        // 2) DeVec
        // 3) uninitialized vector
        if (vec_type == GrB_SPARSE) {
            assignSparse(&w->sparse_, mask, accum, val, indices, nindices,
                               desc);
        } else if (vec_type == GrB_DENSE) {
            assignDense(&w->dense_, mask, accum, val, indices, nindices,
                              desc);
        } else {
            //TODO
        }

        if (desc->debug()) {
            std::cout << "===End assign===\n";
            CHECK(w->print());
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
        CHECK(u_t->setStorage(GrB_DENSE));
        CHECK(w->setStorage(GrB_DENSE));

        scatterIndexed(&w->dense_, mask, accum, u->dense_.d_val_, indices, nindices,
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

        if (desc->debug()) {
            std::cout << "===Begin reduce===\n";
            CHECK(u_t->print());
        }

        // Get storage:
        Storage vec_type;
        CHECK(u->getStorage(&vec_type));

        // 2 cases:
        // 1) SpVec
        // 2) DeVec
        if (vec_type == GrB_SPARSE)
            CHECK(reduceInner(val, accum, op, &u->sparse_, desc));
        else if (vec_type == GrB_DENSE)
            CHECK(reduceInner(val, accum, op, &u->dense_, desc));
        else
            return GrB_UNINITIALIZED_OBJECT;

        if (desc->debug()) {
            std::cout << "===End reduce===\n";
            std::cout << "Output: " << *val << std::endl;
        }
        return GrB_SUCCESS;
    }
}
}
#endif //GB_KUN_OPERATIONS_H
