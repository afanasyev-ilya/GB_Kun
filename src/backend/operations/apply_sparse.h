#ifndef GB_KUN_APPLY_SPARSE_H
#define GB_KUN_APPLY_SPARSE_H
#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "../vector/dense_vector/dense_vector.h"
#include "../vector/sparse_vector/sparse_vector.h"
namespace lablas{
namespace backend {

        template <typename U, typename W, typename M,
        typename BinaryOpT, typename UnaryOpT>
        LA_Info applySparse(SparseVector<W>* w,
                         const Vector<M>* mask,
                         BinaryOpT        accum,
                         UnaryOpT         op,
                         SparseVector<U>* u,
                         Descriptor*      desc) {
            VNT vec_size;
            u->get_nnz(&vec_size);
            std::string accum_type = typeid(accum).name();
            bool use_accum = accum_type.size() > 1;

            for (Index i = 0; i < vec_size; ++i) {
                if (use_accum) {
                    w->get_vals()[i] = op(u->get_vals()[i]);
                } else {
                    w->get_vals()[i] = op(u->get_vals()[i]);
                }
            }
            return GrB_SUCCESS;
        }

}
}
#endif //GB_KUN_APPLY_SPARSE_H
