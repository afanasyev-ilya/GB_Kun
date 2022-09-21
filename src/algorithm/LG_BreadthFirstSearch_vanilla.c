//------------------------------------------------------------------------------
// LG_BreadthFirstSearch_vanilla:  BFS using only GraphBLAS API
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// References:
//
// Contribute by Scott McMillan, derived from examples in the appendix of
// The GraphBLAS C API Specification, v1.3.0

#define LAGraph_FREE_WORK   \
{                           \
    GrB_free (&frontier);   \
}

#define LAGraph_FREE_ALL    \
{                           \
    LAGraph_FREE_WORK ;     \
    GrB_free (&l_parent);   \
    GrB_free (&l_level);    \
}

#include "LG_internal.h"

#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)       \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
/*printf("matrix has %ld\n edges", nvals);*/                                            \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                           \
double my_perf = my_nvals * 2.0 / ((my_t2 - my_t1)*1e9);                                         \
double my_bw = my_nvals * bytes_per_flop/((my_t2 - my_t1)*1e9);                                  \
/*printf("edges: %lf\n", nvals);*/                                                   \
printf("%s time %lf (ms)\n", op_name, (my_t2-my_t1)*1000);                           \
/*printf("%s perf %lf (GFLop/s)\n", op_name, perf);*/                                \
/*printf("%s BW %lf (GB/s)\n", op_name, bw);*/                                       \
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");                                                 \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);                                                                           \


//****************************************************************************
int LG_BreadthFirstSearch_vanilla
(
    GrB_Vector    *level,
    GrB_Vector    *parent,
    LAGraph_Graph  G,
    GrB_Index      src,
    bool           pushpull,
    char          *msg
)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector frontier = NULL;     // the current frontier
    GrB_Vector l_parent = NULL;     // parent vector
    GrB_Vector l_level = NULL;      // level vector

    bool compute_level  = (level != NULL);
    bool compute_parent = (parent != NULL);
    if (compute_level ) (*level ) = NULL;
    if (compute_parent) (*parent) = NULL;

    LG_CHECK (LAGraph_CheckGraph (G, msg), -101, "graph is invalid") ;

    if (!(compute_level || compute_parent))
    {
        // nothing to do
        return (0) ;
    }

    //--------------------------------------------------------------------------
    // get the problem size and properties
    //--------------------------------------------------------------------------
    GrB_Matrix A = G->A ;

    GrB_Index n;
    GrB_TRY( GrB_Matrix_nrows (&n, A) );
    LG_CHECK( src >= n, -102, "src is out of range") ;

    GrB_Matrix AT ;
    GrB_Vector Degree = G->rowdegree ;
    LAGraph_Kind kind = G->kind ;

    if (kind == LAGRAPH_ADJACENCY_UNDIRECTED ||
       (kind == LAGRAPH_ADJACENCY_DIRECTED &&
        G->A_structure_is_symmetric == LAGRAPH_TRUE))
    {
        // AT and A have the same structure and can be used in both directions
        AT = G->A ;
    }
    else
    {
        // AT = A' is different from A
        AT = G->AT ;
    }

    // determine the semiring type
    GrB_Type     int_type  = (n > INT32_MAX) ? GrB_INT64 : GrB_INT32 ;
    GrB_BinaryOp second_op = (n > INT32_MAX) ? GrB_SECOND_INT64 : GrB_SECOND_INT32;
    GrB_Semiring semiring  = NULL;
    GrB_IndexUnaryOp ramp = NULL ;

    if (compute_parent)
    {
        // create the parent vector.  l_parent(i) is the parent id of node i
        GrB_TRY (GrB_Vector_new(&l_parent, int_type, n)) ;

        semiring = (n > INT32_MAX) ?
            GrB_MIN_FIRST_SEMIRING_INT64 : GrB_MIN_FIRST_SEMIRING_INT32;

        // create a sparse integer vector frontier, and set frontier(src) = src
        GrB_TRY (GrB_Vector_new(&frontier, int_type, n)) ;
        GrB_TRY (GrB_Vector_setElement(frontier, src, src)) ;

        // pick the ramp operator
        ramp = (n > INT32_MAX) ? GrB_ROWINDEX_INT64 : GrB_ROWINDEX_INT32 ;
    }
    else
    {
        // only the level is needed
        semiring = LAGraph_structural_bool ;

        // create a sparse boolean vector frontier, and set frontier(src) = true
        GrB_TRY (GrB_Vector_new(&frontier, GrB_BOOL, n)) ;
        GrB_TRY (GrB_Vector_setElement(frontier, true, src)) ;
    }

    if (compute_level)
    {
        // create the level vector. v(i) is the level of node i
        // v (src) = 0 denotes the source node
        GrB_TRY (GrB_Vector_new(&l_level, int_type, n)) ;
        //GrB_TRY (GrB_Vector_setElement(l_level, 0, src)) ;
    }

    //--------------------------------------------------------------------------
    // BFS traversal and label the nodes
    //--------------------------------------------------------------------------
    GrB_Index nq = 1 ;          // number of nodes in the current level
    GrB_Index last_nq = 0 ;
    GrB_Index current_level = 0;
    GrB_Index nvals = 1;

    // {!mask} is the set of unvisited nodes
    GrB_Vector mask = (compute_parent) ? l_parent : l_level ;

    // parent BFS
    do
    {
        if (compute_level)
        {
            // assign levels: l_level<s(frontier)> = current_level
            GrB_TRY( GrB_assign(l_level, frontier, GrB_NULL,
                                current_level, GrB_ALL, n, GrB_DESC_S) );
            ++current_level;
        }

        if (compute_parent)
        {
            // frontier(i) currently contains the parent id of node i in tree.
            // l_parent<s(frontier)> = frontier
            GrB_TRY( GrB_assign(l_parent, frontier, GrB_NULL,
                                frontier, GrB_ALL, n, GrB_DESC_S) );

            // convert all stored values in frontier to their indices
            GrB_TRY (GrB_apply (frontier, GrB_NULL, GrB_NULL, ramp,
                frontier, 0, GrB_NULL)) ;
        }

        // frontier = kth level of the BFS
        // mask is l_parent if computing parent, l_level if computing just level
        GrB_TRY( GrB_vxm(frontier, mask, GrB_NULL, semiring,
                         frontier, A, GrB_DESC_RSC));

        // done if frontier is empty
        GrB_TRY( GrB_Vector_nvals(&nvals, frontier) );
    } while (nvals > 0);

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (compute_parent) (*parent) = l_parent ;
    if (compute_level ) (*level ) = l_level ;
    LAGraph_FREE_WORK ;
    return (0) ;
}
