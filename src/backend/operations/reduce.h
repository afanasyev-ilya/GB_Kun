#ifndef GB_KUN_REDUCE_H
#define GB_KUN_REDUCE_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
namespace lablas {
namespace backend{

    /*TODO - maybe reduce two reduces into 1 func? */

// Dense vector variant
template <typename T, typename U,
typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const DenseVector<U>*  u,
                 Descriptor*            desc) {

    //TODO via monoid
    T temp_val = 0;
    VNT vec_size;
    u->get_size(&vec_size);
    for (int i = 0; i < vec_size; i++) {
        temp_val += u->get_vals()[i];
    }
    if (accum != NULL) {
        *val = *val + temp_val;
    } else {
        *val = temp_val;
    }
}

// Sparse vector variant
template <typename T, typename U,
        typename BinaryOpT, typename MonoidT>
        Info reduceInner(T*                     val,
                         BinaryOpT              accum,
                         MonoidT                op,
                         const SparseVector<U>* u,
                         Descriptor*            desc) {
            //TODO via monoid
            T temp_val = 0;
            VNT vec_size;
            u->get_size(&vec_size);
            for (int i = 0; i < vec_size; i++) {
                temp_val += u->get_vals()[i];
            }
            if (accum != NULL) {
                *val = *val + temp_val;
            } else {
                *val = temp_val;
            }

        }
}
}
#endif //GB_KUN_REDUCE_H
