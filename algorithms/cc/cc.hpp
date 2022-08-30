/**
  @file cc.hpp
  @author S.krymskiy
  @version Revision 1.1
  @brief CC algorithm.
  @details  Code is based on the algorithm described in the following paper.
  Zhang, Azad, Hu. FastSV: FastSV: A Distributed-Memory Connected Component
  Algorithm with Fast Convergence (SIAM PP20).
  @date June 12, 2022
*/

#pragma once
#include "../../src/gb_kun.h"
#include "../../src/cpp_graphblas/types.hpp"
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)


//! Lablas namespace
namespace lablas {
//! Algorithm namespace
    namespace algorithm {

/**
 * Connected components algorithm (FastSV variant) implemented with GraphBLAS primitives
 * @brief The function implements the CC algorithm in notation of GraphBLAS standard
 * @param v Vector to store component labels
 * @param A Target matrix representing an input graph
 * @param seed seed?
 * @param desc Descriptor to store some auxilary data
*/

void cc(Vector<int>*       v,
        const Matrix<int> *A,
        int seed,
        Descriptor *desc)
{
    double loop_dup_time = 0;
    double loop_mxv_time = 0;
    double loop_ewiseadd_time = 0;
    double loop_assignscatter_time = 0;
    double loop_extract_time = 0;
    double loop_ewisemult_time = 0;
    double loop_reduce_time = 0;
    double loop_assign_time = 0;
    double loop_desc_time = 0;

    double t_op_start;

    const auto t1 = omp_get_wtime();

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

    Index succ = 0;
    float gpu_tight_time = 0.f;
    Index niter = 100;
    Index iter = 0;

    const auto t2 = omp_get_wtime();

    for (iter = 1; iter <= niter; ++iter) {
        // Duplicate parent.

        t_op_start = omp_get_wtime();
        parent_temp.dup(&parent);
        loop_dup_time += omp_get_wtime() - t_op_start;

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        t_op_start = omp_get_wtime();
        mxv(&min_neighbor_parent_temp, MASK_NULL, GrB_NULL,
            MinimumSelectSecondSemiring<int>(), A, &grandparent, desc);
        loop_mxv_time += omp_get_wtime() - t_op_start;

        //cout << "min_neighbor_parent_temp: ";
        //min_neighbor_parent_temp.print();

        t_op_start = omp_get_wtime();
        eWiseAdd(&min_neighbor_parent, MASK_NULL, GrB_NULL,
                 MinimumSelectSecondSemiring<int>(), &min_neighbor_parent, &min_neighbor_parent_temp, desc);
        loop_ewiseadd_time += omp_get_wtime() - t_op_start;

        //cout << "min_neighbor_paren: ";
        //min_neighbor_parent.print();

        // f[f[u]] = mngf[u]. Second does nothing (imitating comma operator)
        t_op_start = omp_get_wtime();
        assignScatter(&parent, MASK_NULL, GrB_NULL,
                      &min_neighbor_parent, &parent_temp, parent_temp.nvals(), desc);
        loop_assignscatter_time += omp_get_wtime() - t_op_start;

        //cout << "after assign: ";
        //parent.print();

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        t_op_start = omp_get_wtime();
        eWiseAdd(&parent, MASK_NULL, GrB_NULL,MinimumPlusSemiring<int>(), &parent, &min_neighbor_parent, desc);
        loop_ewiseadd_time += omp_get_wtime() - t_op_start;

        //cout << "after hooking: ";
        //parent.print();

        // 3) Shortcutting.
        // f = min(f, gf)
        t_op_start = omp_get_wtime();
        eWiseAdd(&parent, MASK_NULL, GrB_NULL, MinimumPlusSemiring<int>(), &parent, &parent_temp, desc);
        loop_ewiseadd_time += omp_get_wtime() - t_op_start;

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        t_op_start = omp_get_wtime();
        extract(&grandparent, MASK_NULL, GrB_NULL, &parent, &parent, desc);
        loop_extract_time += omp_get_wtime() - t_op_start;

        // 5) Check termination.
        t_op_start = omp_get_wtime();
        eWiseMult(&diff, MASK_NULL, GrB_NULL,
                  MinimumNotEqualToSemiring<int, int, bool>(), &grandparent_temp, &grandparent, desc);
        loop_ewisemult_time += omp_get_wtime() - t_op_start;
        t_op_start = omp_get_wtime();
        reduce<Index, bool>(&succ, GrB_NULL, PlusMonoid<Index>(), &diff, desc);
        loop_reduce_time += omp_get_wtime() - t_op_start;
        #ifdef __DEBUG_INFO__
        cout << "succ: " << succ << endl;
        #endif
        if (succ == 0)
        {
            break;
        }
        t_op_start = omp_get_wtime();
        grandparent_temp.dup(&grandparent);
        loop_dup_time += omp_get_wtime() - t_op_start;

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        t_op_start = omp_get_wtime();
        desc->toggle(GrB_MASK);
        loop_desc_time += omp_get_wtime() - t_op_start;
        t_op_start = omp_get_wtime();
        assign(&grandparent, &diff, nullptr,
                                    std::numeric_limits<int>::max(), GrB_ALL, A_nrows, desc);
        loop_assign_time += omp_get_wtime() - t_op_start;
        t_op_start = omp_get_wtime();
        desc->toggle(GrB_MASK);
        Desc_value a;
        desc->get(GrB_MASK, &a);
        loop_desc_time += omp_get_wtime() - t_op_start;
    }

    const auto t3 = omp_get_wtime();

    v->dup(&parent);
    std::cout << "Did " << iter <<  " iterations" << std::endl;

    const auto t4 = omp_get_wtime();

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_before_loop", (t2 - t1) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop", (t3 - t2) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_after_loop", (t4 - t3) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_dup_time", loop_dup_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_mxv_time", loop_mxv_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_ewiseadd_time", loop_ewiseadd_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_assignscatter_time", loop_assignscatter_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_extract_time", loop_extract_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_ewisemult_time", loop_ewisemult_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_reduce_time", loop_reduce_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_assign_time", loop_assign_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "cc_loop_desc_time", loop_desc_time * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

}

}  // namespace algorithm
}  // namespace graphblas