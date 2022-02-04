#pragma once
#include "../../src/gb_kun.h"
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

namespace lablas {
namespace algorithm {

// Code is based on the algorithm described in the following paper.
// Zhang, Azad, Hu. FastSV: FastSV: A Distributed-Memory Connected Component
// Algorithm with Fast Convergence (SIAM PP20).
float cc(Vector<int>*       v,
         const Matrix<int>* A,
         int                seed,
         Descriptor*        desc) {
    Index A_nrows;

    // Difference vector.
    Vector<bool> diff(A_nrows);

    // Parent vector.
    // f in Zhang paper.
    Vector<int> parent(A_nrows);
    Vector<int> parent_temp(A_nrows);

    // Grandparent vector.
    // gf in Zhang paper.
    Vector<int> grandparent(A_nrows);
    Vector<int> grandparent_temp(A_nrows);

    // Min neighbor grandparent vector.
    // mngf in Zhang paper.
    Vector<int> min_neighbor_parent(A_nrows);
    Vector<int> min_neighbor_parent_temp(A_nrows);

    // Initialize parent and min_neighbor_parent to:
    // [0]:0 [1]:1 [2]:2 [3]:3 [4]:4, etc.
    parent.fillAscending(A_nrows);
    min_neighbor_parent.dup(&parent);
    min_neighbor_parent_temp.dup(&parent);
    grandparent.dup(&parent);
    grandparent_temp.dup(&parent);

    int iter = 1;
    int succ = 0;
    float gpu_tight_time = 0.f;
    int niter = 10;


    for (iter = 1; iter <= niter; ++iter) {

        // Duplicate parent.
        parent_temp.dup(&parent);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        mxv(&min_neighbor_parent_temp, MASK_NULL, GrB_NULL,
                                MinimumSelectSecondSemiring<int>(), A, &grandparent, desc);
        eWiseAdd(&min_neighbor_parent, MASK_NULL, GrB_NULL,
                                      MinimumSelectSecondSemiring<int>(), &min_neighbor_parent,
                                      &min_neighbor_parent_temp, desc);
        // f[f[u]] = mngf[u]
        assignScatter(&parent, MASK_NULL, GrB_NULL,
                                           &min_neighbor_parent, &parent_temp, parent_temp.nvals(), desc);

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        eWiseAdd(&parent, MASK_NULL, GrB_NULL,
                                      MinimumPlusSemiring<int>(), &parent, &min_neighbor_parent, desc);

        // 3) Shortcutting.
        // f = min(f, gf)
        eWiseAdd(&parent, MASK_NULL, GrB_NULL,
                                      MinimumPlusSemiring<int>(), &parent, &parent_temp, desc);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        extract(&grandparent, MASK_NULL, GrB_NULL,
                                           &parent, &parent, desc);

        // 5) Check termination.
        eWiseMult(&diff, MASK_NULL, GrB_NULL,
                                        MinimumNotEqualToSemiring<int, int, bool>(), &grandparent_temp,
                                        &grandparent, desc);
        reduce<int, bool>(&succ, GrB_NULL, PlusMonoid<int>(), &diff, desc);
        if (succ == 0) {
            break;
        }
        grandparent_temp.dup(&grandparent);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        CHECK(desc->toggle(GrB_MASK));
        assign(&grandparent, &diff, GrB_NULL,
                                    std::numeric_limits<int>::max(), GrB_ALL, A_nrows, desc);
        CHECK(desc->toggle(GrB_MASK));

    }

//    // Copy result to output.
//    CHECK(v->dup(&parent));
//    if (desc->descriptor_.timing_ > 0) {
//        gpu_tight.Stop();
//        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
//                "push" : "pull";
//        if (desc->descriptor_.timing_ == 2)
//            std::cout << iter << ", " << succ << "/" << A_nrows << ", "
//            << vxm_mode << ", " << gpu_tight.ElapsedMillis() << "\n";
//        gpu_tight_time += gpu_tight.ElapsedMillis();
//        return gpu_tight_time;
//    }
    return 0.f;
}

template <typename a>
int ccCpu(Index             seed,
          Matrix<a>*        A,
          std::vector<int>* h_cc_cpu) {
    SimpleReferenceCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
                      A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, seed);
}

template <typename a>
int verifyCc(const Matrix<a>*        A,
             const std::vector<int>& h_cc_cpu,
             bool                    suppress_zero = false) {
    SimpleVerifyCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
                   A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, suppress_zero);
}
}  // namespace algorithm
}  // namespace graphblas