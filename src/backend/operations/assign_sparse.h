#ifndef GB_KUN_ASSIGN_SPARSE_H
#define GB_KUN_ASSIGN_SPARSE_H


#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"

namespace lablas{
namespace backend {


template <typename W, typename T, typename M, typename I,
        typename BinaryOpT>
    LA_Info assignSparse(SparseVector<W>*  w,
                         Vector<M>*       mask,
                         BinaryOpT        accum,
                         T                val,
                         const Vector<I>* indices,
                         Index            nindices,
                         Descriptor*      desc) {
        VNT vec_nz;
        w->get_nz(&vec_nz);

        /* Let us assume mask is dense */

        for (int i = 0; i < vec_nz; i++) {
            VNT idx = w->get_ids()[i];
            if (mask->getDense()[i] != 0) {
                w->get_vals()[idx] = val;
            }
        }
    }

}
}




#endif //GB_KUN_ASSIGN_SPARSE_H
