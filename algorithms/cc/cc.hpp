#pragma once
#include "../../src/gb_kun.h"
#include "../../src/cpp_graphblas/types.hpp"
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

    if(!A->is_symmetric()) {
        std::cout << "Matrix is not symmetric, aborting CC..." << std::endl;
        return 1;
    };
    Index A_nrows;
    A->get_nrows(&A_nrows);
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
//    parent.print();
    min_neighbor_parent.dup(&parent);
//    min_neighbor_parent.print();
    min_neighbor_parent_temp.dup(&parent);
//    min_neighbor_parent_temp.print();
    grandparent.dup(&parent);
//    grandparent.print();
    grandparent_temp.dup(&parent);
//    grandparent_temp.print();

    int succ = 0;
    float gpu_tight_time = 0.f;
    int niter = 100;

    for (int iter = 1; iter <= niter; ++iter) {
        // Duplicate parent.
        parent_temp.dup(&parent);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        mxv(&min_neighbor_parent_temp, MASK_NULL, GrB_NULL,
                                MinimumSelectSecondSemiring<int>(), A, &grandparent, desc);

        //cout << "min_neighbor_parent_temp: ";
        //min_neighbor_parent_temp.print();

        eWiseAdd(&min_neighbor_parent, MASK_NULL, GrB_NULL,
                 MinimumSelectSecondSemiring<int>(), &min_neighbor_parent, &min_neighbor_parent_temp, desc);

        //cout << "min_neighbor_paren: ";
        //min_neighbor_parent.print();

        // f[f[u]] = mngf[u]. Second does nothing (imitating comma operator)
        assignScatter(&parent, MASK_NULL, GrB_NULL,
                      &min_neighbor_parent, &parent_temp, parent_temp.nvals(), desc);

        //cout << "after assign: ";
        //parent.print();

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        eWiseAdd(&parent, MASK_NULL, GrB_NULL,MinimumPlusSemiring<int>(), &parent, &min_neighbor_parent, desc);

        //cout << "after hooking: ";
        //parent.print();

        // 3) Shortcutting.
        // f = min(f, gf)
        eWiseAdd(&parent, MASK_NULL, GrB_NULL, MinimumPlusSemiring<int>(), &parent, &parent_temp, desc);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        extract(&grandparent, MASK_NULL, GrB_NULL, &parent, &parent, desc);

        // 5) Check termination.
        eWiseMult(&diff, MASK_NULL, GrB_NULL,
                  MinimumNotEqualToSemiring<int, int, bool>(), &grandparent_temp, &grandparent, desc);
        reduce<int, bool>(&succ, GrB_NULL, PlusMonoid<int>(), &diff, desc);
        #ifdef __DEBUG_INFO__
        cout << "succ: " << succ << endl;
        #endif
        if (succ == 0)
        {
            break;
        }
        grandparent_temp.dup(&grandparent);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        desc->toggle(GrB_MASK);
        assign(&grandparent, &diff, nullptr,
                                    std::numeric_limits<int>::max(), GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);
        Desc_value a;
        desc->get(GrB_MASK, &a);
    }
    v->dup(&parent);

    return 0.f;
}

}  // namespace algorithm
}  // namespace graphblas