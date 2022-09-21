//------------------------------------------------------------------------------
// LAGraph_AllKTruss.c: find all k-trusses of a graph
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// LAGraph_AllKTruss: find all k-trusses of a graph via GraphBLAS.
// Contributed by Tim Davis, Texas A&M.

// Given a symmetric graph A with no-self edges, LAGraph_AllKTruss finds all
// k-trusses of A.

// The output matrices Cset [3..kmax-1] are the k-trusses of A.  Their edges
// are a subset of A.  Each edge in C = Cset [k] is part of at least k-2
// triangles in C.  The structure of C is the adjacency matrix of the k-truss
// subgraph of A.  The edge weights of C are the support of each edge.  That
// is, C(i,j)=nt if the edge (i,j) is part of nt triangles in C.  All edges in
// C have support of at least k-2.  The total number of triangles in C is
// sum(C)/6.  The number of edges in C is nnz(C)/2.  C = Cset [k] is returned
// as symmetric with a zero-free diagonal.  Cset [kmax] is an empty matrix
// since the kmax-truss is empty.

// The arrays ntris, nedges, and nstepss hold the output statistics.
// ntris   [k] = # of triangles in the k-truss
// nedges  [k] = # of edges in the k-truss
// nstepss [k] = # of steps required to compute the k-truss

// Usage: constructs all k-trusses of A, for k = 3:kmax

//      int64_t kmax ;
//      GrB_Matrix_nrows (&n, A) ;
//      int64_t n4 = (n > 4) ? n : 4 ;
//      GrB_Matrix *Cset = LAGraph_malloc (n4, sizeof (GrB_Matrix)) ;
//      int64_t *ntris   = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int64_t *nedges  = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int64_t *nstepss = LAGraph_malloc (n4, sizeof (int64_t)) ;
//      int result = LAGraph_AllKTruss (&Cset, &kmax, ntris, nedges,
//          nstepss, G, msg) ;

// TODO: add experimental/benchmark/ktruss_demo.c to benchmark k-truss
// and all-k-truss

// TODO: consider LAGraph_KTrussNext to compute the (k+1)-truss from the
// k-truss

#define LAGraph_FREE_ALL                    \
{                                           \
    for (int64_t kk = 3 ; kk <= k ; kk++)   \
    {                                       \
        GrB_free (&(Cset [kk])) ;           \
    }                                       \
}

#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// C = LAGraph_AllKTruss: find all k-trusses a graph
//------------------------------------------------------------------------------

int LAGraph_AllKTruss   // compute all k-trusses of a graph
(
    // outputs
    GrB_Matrix *Cset,   // size n, output k-truss subgraphs
    int64_t *kmax,      // smallest k where k-truss is empty
    int64_t *ntris,     // size max(n,4), ntris [k] is #triangles in k-truss
    int64_t *nedges,    // size max(n,4), nedges [k] is #edges in k-truss
    int64_t *nstepss,   // size max(n,4), nstepss [k] is #steps for k-truss
    // input
    LAGraph_Graph G,    // input graph
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    int64_t k = 0 ;
    LG_CHECK (Cset == NULL || nstepss == NULL || kmax == NULL || ntris == NULL
        || nedges == NULL, GrB_NULL_POINTER, "input(s) are NULL") ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), GrB_INVALID_OBJECT,
        "graph is invalid") ;

    if (G->kind == LAGRAPH_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGRAPH_ADJACENCY_DIRECTED &&
        G->A_structure_is_symmetric == LAGRAPH_TRUE))
    {
        // the structure of A is known to be symmetric
        ;
    }
    else
    {
        // A is not known to be symmetric
        LG_CHECK (true, -1005, "G->A must be symmetric") ;
    }

    // no self edges can be present
    LG_CHECK (G->ndiag != 0, -1004, "G->ndiag must be zero") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    for (k = 0 ; k <= 3 ; k++)
    {
        Cset [k] = NULL ;
        ntris   [k] = 0 ;
        nedges  [k] = 0 ;
        nstepss [k] = 0 ;
    }
    k = 3 ;
    (*kmax) = 0 ;

    //--------------------------------------------------------------------------
    // initialzations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GrB_Matrix S = G->A ;
    GrB_TRY (GrB_Matrix_nrows (&n, S)) ;
    GrB_TRY (GrB_Matrix_new (&(Cset [k]), GrB_UINT32, n, n)) ;
    GrB_Matrix C = Cset [k] ;
    GrB_Index nvals, nvals_last ;
    GrB_TRY (GrB_Matrix_nvals (&nvals_last, S)) ;
    int64_t nsteps = 0 ;

    //--------------------------------------------------------------------------
    // find all k-trusses
    //--------------------------------------------------------------------------

    while (true)
    {
        // C{S} = S*S'
        GrB_TRY (GrB_mxm (C, S, NULL, LAGraph_plus_one_uint32, S, S,
            GrB_DESC_RST1)) ;
        // keep entries in C that are >= k-2
        GrB_TRY (GrB_select (C, NULL, NULL, GrB_VALUEGE_UINT32, C, k-2, NULL)) ;
        nsteps++ ;
        // check if k-truss has been found
        GrB_TRY (GrB_Matrix_nvals (&nvals, C)) ;
        if (nvals == nvals_last)
        {
            // k-truss has been found
            int64_t nt = 0 ;
            GrB_TRY (GrB_reduce (&nt, NULL, GrB_PLUS_MONOID_INT64, C, NULL)) ;
            ntris   [k] = nt / 6 ;
            nedges  [k] = nvals / 2 ;
            nstepss [k] = nsteps ;
            nsteps = 0 ;
            if (nvals == 0)
            {
                // this is the last k-truss
                (*kmax) = k ;
                return (GrB_SUCCESS) ;
            }
            S = C ;             // S = current k-truss for k+1 iteration
            k++ ;               // advance to the next k-tryss
            GrB_TRY (GrB_Matrix_new (&(Cset [k]), GrB_UINT32, n, n)) ;
            C = Cset [k] ;      // C = new matrix for next k-truss
        }
        else
        {
            // advance to the next step, still computing the current k-truss
            nvals_last = nvals ;
            S = C ;
        }
    }
}

