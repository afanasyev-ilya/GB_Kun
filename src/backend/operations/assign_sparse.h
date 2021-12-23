#ifndef GB_KUN_ASSIGN_SPARSE_H
#define GB_KUN_ASSIGN_SPARSE_H


#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

namespace lablas{
namespace backend {


template <typename W, typename T, typename M,
    typename BinaryOpT>
LA_Info assignSparse(SparseVector<W>*  w,
                     const Vector<M>*       mask,
                     BinaryOpT        accum,
                     T                val,
                     const Index*     indices,
                     Index            nindices,
                     Descriptor*      desc) {
    VNT vec_size;
    w->get_nnz(&vec_size);
    std::string accum_type = typeid(accum).name();

    if(indices == NULL)
    {
        if (mask != NULL) {
            for (int i = 0; i < vec_size; i++) {
                if (mask->getDense()->get_vals()[i] != 0) {
                    if (accum_type.size() <= 1) {
                        w->get_vals()[i] = val;
                    } else {
                        w->get_vals()[i] = val;
                    }
                }
            }
        } else {
            for (int i = 0; i < vec_size; i++) {
                if (accum_type.size() <= 1) {
                    w->get_vals()[i] = val;
                } else {
                    w->get_vals()[i] = val;
                }
            }
        }
    }
    else
    {
        throw "Error in assignSparse: indices != NULL not supported";
    }

    return GrB_SUCCESS;
}

}
}




#endif //GB_KUN_ASSIGN_SPARSE_H
