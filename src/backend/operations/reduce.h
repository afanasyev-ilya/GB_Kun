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
LA_Info reduceInner(T*                    val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const DenseVector<U>*  u,
                 Descriptor*            desc) {

    T temp_val = 0;
    std::string accum_type = typeid(accum).name();
    std::string op_type =    typeid(op).name();

    VNT vec_size;
    u->get_size(&vec_size);
    const U* sparse_pointer = u->get_vals();

#pragma om parallel for
    for (int i = 0; i < vec_size; i++) {
        U value  = sparse_pointer[i];
#pragma omp critical
        {

            temp_val = op(temp_val, value);
        }
    }

    /*in accum case, use temp buffer var */
    if (accum_type.size() > 1) {
        //            auto add_op = extractAdd(op);
        *val = temp_val;
    } else {
        *val = temp_val;
    }

    return GrB_SUCCESS;
}

// Sparse vector variant
template <typename T, typename U,
        typename BinaryOpT, typename MonoidT>
    LA_Info reduceInner(T*                     val,
                        BinaryOpT              accum,
                        MonoidT                op,
                        const SparseVector<U>* u,
                        Descriptor*            desc) {

        T temp_val = 0;
        std::string accum_type = typeid(accum).name();
        std::string op_type =    typeid(op).name();

        VNT vec_size;
        u->get_nnz(&vec_size);
        const U* sparse_pointer = u->get_vals();

#pragma om parallel for
        for (int i = 0; i < vec_size; i++) {
            U value  = sparse_pointer[i];
#pragma omp critical
        {

                temp_val = op(temp_val, value);
        }
        }

        /*in accum case, use temp buffer var */
        if (accum_type.size() > 1) {
//            auto add_op = extractAdd(op);
            *val = temp_val;
        } else {
            *val = temp_val;
        }

        return GrB_SUCCESS;

    }
}
}
#endif //GB_KUN_REDUCE_H
