#ifndef GB_KUN_SCATTER_H
#define GB_KUN_SCATTER_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
namespace lablas{
namespace backend{


// Dense vector indexed variant
template <typename W, typename M, typename U, typename I,
        typename BinaryOpT>
LA_Info scatterIndexed(DenseVector<W>*  w,
                    const Vector<M>* mask,
                    BinaryOpT        accum,
                    U*               d_u_val,
                    I*               d_indices,
                    Index            nindices,
                    Descriptor*      desc) {
    int w_size;
    w->get_size(&w_size);

    if (d_indices == NULL) {
        for (VNT i = 0; i < w_size; i++) {
            U val = d_u_val[i];
            Index ind = i;
            w->get_vals()[ind] = val;
        }
    } else {
        for (VNT i = 0; i < w_size; i++) {
            U val = d_u_val[i];
            Index ind = d_indices[i];
            if (ind >= 0 && ind < w_size) {
                w->get_vals()[ind] = val;
            }
        }
    }

    return GrB_SUCCESS;
}

}
}
#endif //GB_KUN_SCATTER_H
