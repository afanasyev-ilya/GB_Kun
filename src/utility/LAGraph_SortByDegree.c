//------------------------------------------------------------------------------
// LAGraph_SortByDegree: sort a graph by its row or column degree
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// Contributed by Tim Davis, Texas A&M University.

//------------------------------------------------------------------------------

// LAGraph_SortByDegree computes a permutation vector P that sorts a graph
// by degree (either row or column degree of its adjacency matrix A).
// If G is undirected, or if G is directed but is known to have a symmetric
// adjacency matrix, then G->rowdegree is used (and byrow is ignored).
// Otherwise, if G->rowdegree is used if byrow is true, and G->coldegree is
// used if byrow is false.

// G->rowdegree or G->coldegree must first be computed.  An error is returned
// if the required degree vector has not yet been computed.  See
// LAGraph_Property_RowDegree and LAGraph_Property_ColDegree.

// The permutation is in ascending order of degree if ascending is true, and
// in descending order otherwise.

// Ties are broken by the node id, so the sort is always predicable.  Lower
// numbered rows/columns always appear before higher ones, if they have the
// same degree.

// The output is a permutation P where P [k] = i if row i is the kth row in
// the permutation (or P [k] = j if column j is the kth column in the
// permutation, with byrow false).

#define LAGraph_FREE_WORK           \
{                                   \
    LAGraph_Free ((void **) &W) ;   \
    LAGraph_Free ((void **) &D) ;   \
}

#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_FREE_WORK ;             \
    LAGraph_Free ((void **) &P) ;   \
}

#include "LG_internal.h"

int LAGraph_SortByDegree    // returns 0 if successful, -1 if failure
(
    // output
    int64_t **P_handle,     // P is returned as a permutation vector of size n
    // input
    LAGraph_Graph G,        // graph of n nodes
    bool byrow,             // if true, sort G->rowdegree, else G->coldegree
    bool ascending,         // sort in ascending or descending order
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    int64_t *P = NULL ;
    int64_t *W = NULL ;
    int64_t *D = NULL ;
    LG_CHECK (P_handle == NULL, -1, "P is null") ;
    (*P_handle) = NULL ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), -1, "graph is invalid") ;

    GrB_Vector Degree ;

    if (G->kind == LAGRAPH_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGRAPH_ADJACENCY_DIRECTED &&
        G->A_structure_is_symmetric == LAGRAPH_TRUE))
    {
        // the structure of A is known to be symmetric
        Degree = G->rowdegree ;
    }
    else
    {
        // A is not known to be symmetric
        Degree = (byrow) ? G->rowdegree : G->coldegree ;
    }

    LG_CHECK (Degree == NULL, -1, "degree property unknown") ;

    //--------------------------------------------------------------------------
    // decide how many threads to use
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GrB_TRY (GrB_Vector_size (&n, Degree)) ;

    #define CHUNK (64*1024)
    int nthreads ;
    LAGraph_TRY (LAGraph_GetNumThreads (&nthreads, msg)) ;
    nthreads = LAGraph_MIN (nthreads, n/CHUNK) ;
    nthreads = LAGraph_MAX (nthreads, 1) ;

    //--------------------------------------------------------------------------
    // allocate result and workspace
    //--------------------------------------------------------------------------

    P = LAGraph_Malloc (n, sizeof (int64_t)) ;
    D = LAGraph_Malloc (n, sizeof (int64_t)) ;
    W = LAGraph_Malloc (2*n, sizeof (int64_t)) ;
    LG_CHECK (D == NULL || P == NULL || W == NULL, -1, "out of memory") ;
    int64_t *W0 = W ;
    int64_t *W1 = W + n ;

    //--------------------------------------------------------------------------
    // construct the pair [D,P] to sort
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t k = 0 ; k < n ; k++)
    {
        D [k] = 0 ;
        P [k] = k ;
    }

    // extract the degrees
    GrB_Index nvals = n ;
    GrB_TRY (GrB_Vector_extractTuples ((GrB_Index *) W0, W1, &nvals, Degree)) ;

    if (ascending)
    {
        // sort [D,P] in ascending order of degree, tie-breaking on P
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            D [W0 [k]] = W1 [k] ;
        }
    }
    else
    {
        // sort [D,P] in descending order of degree, tie-breaking on P
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            D [W0 [k]] = -W1 [k] ;
        }
    }

    LAGraph_Free ((void **) &W) ;

    //--------------------------------------------------------------------------
    // sort by degrees, with ties by node id
    //--------------------------------------------------------------------------

    LAGraph_TRY (LAGraph_Sort2 (D, P, n, nthreads, msg)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LAGraph_FREE_WORK ;
    (*P_handle) = P ;
    return (0) ;
}
