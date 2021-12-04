#ifndef GB_KUN_ASSIGN_DENSE_H
#define GB_KUN_ASSIGN_DENSE_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

namespace lablas{
namespace backend {


template <typename W, typename T, typename M, typename I,
    typename BinaryOpT>
    LA_Info assignDense(DenseVector<W>*  w,
                        const Vector<M>*       mask,
                        BinaryOpT        accum,
                        T                val,
                        const Vector<I>* indices,
                        Index            nindices,
                        Descriptor*      desc) {
        VNT vec_size;
        w->get_size(&vec_size);
        std::string accum_type = typeid(accum).name();
        if (mask != NULL) {
            for (int i = 0; i < vec_size; i++) {
                if (mask->getDense()->get_vals()[i] != 0) {
                    if (accum_type.size() <= 1) {
                        w->get_vals()[i] = val;
                    } else {
                        w->get_vals()[i] =  val;
                    }
                }
            }
        } else {
            for (int i = 0; i < vec_size; i++) {
                    if (accum_type.size() <= 1) {
                        w->get_vals()[i] = val;
                    } else {
                        w->get_vals()[i] =  val;
                    }
            }
        }
        return GrB_SUCCESS;
    }

}
}



#endif //GB_KUN_ASSIGN_DENSE_H
