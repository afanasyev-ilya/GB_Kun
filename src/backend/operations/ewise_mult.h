#ifndef GB_KUN_EWISE_MULT_H
#define GB_KUN_EWISE_MULT_H

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "../vector/dense_vector/dense_vector.h"
#include "../vector/sparse_vector/sparse_vector.h"
namespace lablas{
namespace backend {


// Sparse x sparse vector
template <typename W, typename U, typename V, typename M,
    typename BinaryOpT,     typename SemiringT>
    LA_Info eWiseMultInner(SparseVector<W>*       w,
                           const Vector<M>*       mask,
                           BinaryOpT              accum,
                           SemiringT              op,
                           const SparseVector<U>* u,
                           const SparseVector<V>* v,
                           Descriptor*            desc) {
        std::cout << "Error: eWiseMult sparse-sparse not implemented yet!\n";
        return GrB_SUCCESS;
    }

// Dense x dense vector (no mask and dense mask)
template <typename W, typename U, typename V, typename M,
    typename BinaryOpT,     typename SemiringT>
    LA_Info eWiseMultInner(DenseVector<W>*       w,
                           const Vector<M>*      mask,
                           BinaryOpT             accum,
                           SemiringT             op,
                           const DenseVector<U>* u,
                           const DenseVector<V>* v,
                           Descriptor*           desc) {
        // Get descriptor parameters for SCMP, REPL
        Desc_value scmp_mode, repl_mode;
        desc->get(GrB_MASK, &scmp_mode);
        desc->get(GrB_OUTP, &repl_mode);

        bool use_mask  = (mask != NULL);

        std::string accum_type = typeid(accum).name();
        std::string op_type =    typeid(op).name();
        // Get descriptor parameters for nthreads
        Desc_value nt_mode;
        desc->get(GrB_NT, &nt_mode);
        const int nt = static_cast<int>(nt_mode);


        DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
        DenseVector<V>* v_t = const_cast<DenseVector<V>*>(v);

        Storage mask_type;

        const DenseVector<M>* mask_dense = mask->getDense();
        int u_size;
        u->get_size(&u_size);
        for (int i = 0; i < u_size; i++) {
            if (mask_dense == NULL || mask_dense->get_vals()[i] != 0) {
                U u_val_t = u->get_vals()[i];
                V v_val_t = v->get_vals()[i];

                if (u_val_t == op.identity() || v_val_t == op.identity()) {
                    if (accum_type.size() > 1) {
                        w->get_vals()[i] = op.identity();
                    } else {
                        w->get_vals()[i] = op.identity();
                    }
                } else {
                    if (accum_type.size() > 1) {
                        w->get_vals()[i] = extractMul(op)(u_val_t, v_val_t);
                    } else {
                        w->get_vals()[i] = extractMul(op)(u_val_t, v_val_t);
                    }
                }

            }
        }

        return GrB_SUCCESS;
    }

// Dense x dense vector (sparse mask)
template <typename W, typename U, typename V, typename M,
    typename BinaryOpT,     typename SemiringT>
    LA_Info eWiseMultInner(SparseVector<W>*       w,
                           const SparseVector<M>* mask,
                           BinaryOpT              accum,
                           SemiringT              op,
                           const DenseVector<U>*  u,
                           const DenseVector<V>*  v,
                           Descriptor*            desc) {
        return GrB_SUCCESS;
        //        // Get descriptor parameters for SCMP, REPL
        //        Desc_value scmp_mode, repl_mode;
        //        CHECK(desc->get(GrB_MASK, &scmp_mode));
        //        CHECK(desc->get(GrB_OUTP, &repl_mode));
        //
        //        std::string accum_type = typeid(accum).name();
        //        // TODO(@ctcyang): add accum and replace support
        //        // -have masked variants as separate kernel
        //        // -have scmp as template parameter
        //        // -accum and replace as parts in flow
        //        bool use_mask  = (mask != NULL);
        //        bool use_accum = (accum_type.size() > 1);
        //        bool use_scmp  = (scmp_mode == GrB_SCMP);
        //        bool use_repl  = (repl_mode == GrB_REPLACE);
        //
        //        if (desc->debug()) {
        //            std::cout << "Executing eWiseMult dense-dense (sparse mask)\n";
        //            printState(use_mask, use_accum, use_scmp, use_repl, 0);
        //        }
        //
        //        // Get descriptor parameters for nthreads
        //        Desc_value nt_mode;
        //        CHECK(desc->get(GrB_NT, &nt_mode));
        //        const int nt = static_cast<int>(nt_mode);
        //
        //        // Get number of elements
        //        Index u_nvals;
        //        u->nvals(&u_nvals);
        //
        //        if (use_mask) {
        //            Index mask_nvals;
        //            mask->nvals(&mask_nvals);
        //
        //            dim3 NT, NB;
        //            NT.x = nt;
        //            NT.y = 1;
        //            NT.z = 1;
        //            NB.x = (mask_nvals + nt - 1) / nt;
        //            NB.y = 1;
        //            NB.z = 1;
        //
        //            eWiseMultKernel<<<NB, NT>>>(w->d_ind_, w->d_val_, mask->d_ind_,
        //                                        mask->d_val_, mask_nvals, NULL, op.identity(), extractMul(op),
        //                                        u->d_val_, v->d_val_);
        //
        //            w->nvals_ = mask_nvals;
        //        } else {
        //            std::cout << "Error: Unmasked eWiseMult dense-dense should not";
        //            std::cout << "generate sparse vector output!\n";
        //        }
        //        w->need_update_ = true;
        //
        //        return GrB_SUCCESS;
    }

}
}
#endif //GB_KUN_EWISE_MULT_H
