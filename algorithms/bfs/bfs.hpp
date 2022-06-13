/*
 * This file uses algorithm implementation from LAGraph, which is available under
 * their custom license. For details, see https://github.com/GraphBLAS/LAGraph/blob/reorg/LICENSE
 * */

/**
  @file bfs.hpp
  @author Lastname:Firstname:A00123456:cscxxxxx
  @version Revision 1.1
  @brief BFS algorithm.
  @details This file uses algorithm implementation from LAGraph, which is available under
  their custom license. For details, see https://github.com/GraphBLAS/LAGraph/blob/reorg/LICENSE.
  @date May 12, 2022
*/

#pragma once
#include "../../src/gb_kun.h"

//! Lablas namespace
namespace lablas {
//! Algorithm namespace
    namespace algorithm {


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GrB_Matrix lablas::Matrix<int>*
#define GrB_Vector lablas::Vector<int>*
#define MASK_NULL static_cast<const lablas::Vector<int>*>(NULL)

/**
 * Implementation of Breadth-First Search algorithm from GraphBLAST.
 * Calculates both levels and parents. Currently is not fully tested and only partially implemented.
 * @brief Brief Top-down BFS algorithm from GraphBLAS.
 * @return Returns integer error code (0 if successfully terminated).
 * @param level Level values for each vertex, main result of BFS (number of hops from source vertex).
 * @param parent Parent values for each vertex, secondary result of BFS (id of vertices, from which current vertex has been reached).
 * @param G Input graph, represented as GraphBLAS matrix.
 * @param src Initial vertex, from which BFS traversal is started.
 * @param pushpull If true, optimized direction-optimizing algorithm is used, otherwise - top-down.
*/

int LG_BreadthFirstSearch_vanilla(GrB_Vector *level,
                                  GrB_Vector *parent,
                                  LAGraph_Graph<int> *G,
                                  GrB_Index src,
                                  bool pushpull)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    lablas::Vector<bool> *frontier = NULL;     // the current frontier
    GrB_Vector l_parent = NULL;     // parent vector
    GrB_Vector l_level = NULL;      // level vector
    lablas::Descriptor desc;

    //--------------------------------------------------------------------------
    // get the problem size and properties
    //--------------------------------------------------------------------------
    GrB_Matrix A = G->A;

    GrB_Index n;
    GrB_TRY( GrB_Matrix_nrows (&n, A) );
    assert(src < n && "invalid source node");

    // only the level is needed

    // create a sparse boolean vector frontier, and set frontier(src) = true
    GrB_TRY (GrB_Vector_new(&frontier, GrB_BOOL, n)) ;
    GrB_TRY (GrB_Vector_setElement(frontier, true, src)) ;
    frontier->print();

    // create the level vector. v(i) is the level of node i
    // v (src) = 0 denotes the source node
    GrB_TRY (GrB_Vector_new(&l_level, GrB_INT32, n)) ;

    //--------------------------------------------------------------------------
    // BFS traversal and label the nodes
    //--------------------------------------------------------------------------
    GrB_Index nq = 1; // number of nodes in the current level
    GrB_Index last_nq = 0;
    GrB_Index current_level = 1;
    GrB_Index nvals = 1;

    // {!mask} is the set of unvisited nodes
    GrB_Vector mask = l_level ;

    // parent BFS
    do
    {
        // assign levels: l_level<s(frontier)> = current_level
        GrB_TRY( GrB_assign(l_level, frontier, NULL, current_level, GrB_ALL, n, GrB_DESC_S) );
        l_level->print();
        ++current_level;

        // frontier = kth level of the BFS
        // mask is l_parent if computing parent, l_level if computing just level
        GrB_TRY( GrB_vxm(frontier, mask, NULL, lablas::LogicalOrAndSemiring<bool>(), frontier, A, GrB_DESC_RSC) );
        frontier->print();

        // done if frontier is empty
        GrB_TRY( GrB_Vector_nvals(&nvals, frontier) );
    } while (nvals > 0);

    //l_level->force_to_dense();

    (*level ) = l_level;
    return (0);
}

#undef GrB_Matrix
#undef GrB_Vector
#undef MASK_NULL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Implementation of Breadth-First Search algorithm from GraphBLAST. It is equivalent of standard top-down BFS algorithm.
 * Calculates levels only (without parents). Currently is the fastest among implemented BFS algorithms.
 * @brief Top-down BFS algorithm from LAGraph.
 * @param v Level values for each vertex, main result of BFS (number of hops from source vertex).
 * @param A Input graph, represented as GraphBLAS matrix.
 * @param s Initial vertex, from which BFS traversal is started.
 * @param desc A pointer to GraphBLAS descriptor, must be preliminary allocated.
*/

template <typename T>
void bfs_blast(Vector<T>*       v,
               const Matrix<T> *A,
               Index s,
               Descriptor *desc)
{
    Index A_nrows = A->nrows();

    // Visited vector (use T for now)
    v->fill((T)0);

    // Frontier vectors (use T for now)
    Vector<T> f1(A_nrows);
    Vector<T> f2(A_nrows);

    Desc_value desc_value;
    desc->get(GrB_MXVMODE, &desc_value);
    if (true/*desc_value == GrB_PULLONLY*/)
    {
        f1.fill((T)0);
        f1.set_element((T)1, s);
    }
    else
    {
        /*std::vector<Index> indices(1, s);
        std::vector<T>  values(1, 1.f);
        CHECK(f1.build(&indices, &values, 1, GrB_NULL));*/
    }

    Index iter = 0;
    T succ = 0.f;
    Index unvisited = A_nrows;
    T gpu_tight_time = 0.f;
    Index max_iters = A_nrows;

    SPMV_TIME = 0;
    SPMSPV_TIME = 0;
    CONVERT_TIME = 0;
    double bfs_time = 0;
    double vxm_time = 0;
    double t1 = omp_get_wtime();
    for (iter = 1; iter <= max_iters; ++iter)
    {
        unvisited -= static_cast<int>(succ);
        assign<T, T, T>(v, &f1, GrB_NULL, iter, GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);
        double t1_in = omp_get_wtime();
        vxm<T, T, T, T>(&f2, v, GrB_NULL, LogicalOrAndSemiring<T>(), &f1, A, GrB_DESC_SC);
        double t2_in = omp_get_wtime();
        vxm_time += t2_in - t1_in;

        desc->toggle(GrB_MASK);

        f2.swap(&f1);
        reduce<T, T>(&succ, GrB_NULL, PlusMonoid<T>(), &f1, desc);

        if (succ == 0)
            break;
    }
    double t2 = omp_get_wtime();
    bfs_time += t2 - t1;
    std::cout << "bfs vxm/full: " << 100.0*(vxm_time / bfs_time) << "%\n";
    std::cout << "     bfs SPMV_TIME/vxm: " << 100.0*(SPMV_TIME / vxm_time) << "%\n";
    std::cout << "     bfs SPMSPV_TIME/vxm: " << 100.0*(SPMSPV_TIME / vxm_time) << "%\n";
    std::cout << "     bfs CONVERT/vxm: " << 100.0*(CONVERT_TIME / vxm_time) << "%\n";
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

