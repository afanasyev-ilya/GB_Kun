//------------------------------------------------------------------------------
// LAGraph_BF_basic: Bellman-Ford method for single source shortest paths
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// LAGraph_BF_basic: Bellman-Ford single source shortest paths, returning just
// the shortest path lengths.  Contributed by Jinhao Chen and Tim Davis, Texas
// A&M.

// LAGraph_BF_basic performs a Bellman-Ford to find out shortest path length
// from given source vertex s in the range of [0, n) on graph given as matrix A
// with size n by n. The sparse matrix A has entry A(i, j) if there is edge from
// vertex i to vertex j with weight w, then A(i, j) = w. Furthermore,
// LAGraph_BF_basic requires A(i, i) = 0 for all 0 <= i < n.

// LAGraph_BF_basic returns GrB_SUCCESS regardless of existence of
// negative-weight cycle. However, the GrB_Vector d(k) (i.e., *pd_output) will
// be NULL when negative-weight cycle detected. Otherwise, the vector d has
// d(k) as the shortest distance from s to k.

//------------------------------------------------------------------------------

#define LAGraph_FREE_ALL   \
{                          \
    GrB_free(&d) ;         \
    GrB_free(&dtmp) ;      \
}

#include "LG_internal.h"
#include <LAGraphX.h>

// Given a n-by-n adjacency matrix A and a source vertex s.
// If there is no negative-weight cycle reachable from s, return the distances
// of shortest paths from s as vector d. Otherwise, return d=NULL if there is
// negative-weight cycle.
// pd_output = &d, where d is a GrB_Vector with d(k) as the shortest distance
// from s to k when no negative-weight cycle detected, otherwise, d = NULL.
// A has zeros on diagonal and weights on corresponding entries of edges
// s is given index for source vertex
GrB_Info LAGraph_BF_basic
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols;
    // tmp vector to store distance vector after n (i.e., V) loops
    GrB_Vector d = NULL, dtmp = NULL;

    LG_CHECK (A == NULL || pd_output == NULL, -1001, "inputs are NULL") ;

    *pd_output = NULL;
    GrB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GrB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
    LG_CHECK (nrows != ncols, -1002, "A must be square") ;
    GrB_Index n = nrows;           // n = # of vertices in graph
    LG_CHECK (s >= n || s < 0, -1003, "invalid source node") ;

    // Initialize distance vector, change the d[s] to 0
    GrB_TRY (GrB_Vector_new(&d, GrB_FP64, n));
    GrB_TRY (GrB_Vector_setElement_FP64(d, 0, s));

    // copy d to dtmp in order to create a same size of vector
    GrB_TRY (GrB_Vector_dup(&dtmp, d));

    int64_t iter = 0;      //number of iterations
    bool same = false;     //variable indicating if d=dtmp

    // terminate when no new path is found or more than n-1 loops
    while (!same && iter < n - 1)
    {

        double tic [2] ;
        LAGraph_Tic(tic, NULL);

        // execute semiring on d and A, and save the result to d
        GrB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, A,
            GrB_NULL));
        LAGRAPH_OK (LAGraph_Vector_IsEqual_type(&same, dtmp, d, GrB_FP64, NULL));
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        iter++;
        double t;
        LAGraph_Toc (&t, tic, NULL );
        GrB_Index dnz ;
        GrB_TRY (GrB_Vector_nvals (&dnz, d)) ;
//      printf ("step %3d time %16.4f sec, nvals %.16g\n", iter, t, (double) dnz);
        fflush (stdout) ;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (!same)
    {
        // execute semiring again to check for negative-weight cycle
        GrB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, A,
            GrB_NULL));
        LAGRAPH_OK (LAGraph_Vector_IsEqual_type(&same, dtmp, d, GrB_FP64, NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        if (!same)
        {
            // printf("A negative-weight cycle found. \n");
            LAGraph_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    (*pd_output) = d;
    d = NULL;
    LAGraph_FREE_ALL;
    return (GrB_SUCCESS) ;
}
