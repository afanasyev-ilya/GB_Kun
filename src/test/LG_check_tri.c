//------------------------------------------------------------------------------
// LG_check_tri: compute the number of triangles in a graph (simple method)
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// A very slow, bare-bones triangle count using a parallel dot-product method.
// Computes the sum(sum((A'*A).*A)), in MATLAB notation, where A is symmetric
// and treated as binary (only the structure is used).  Diagonal entries are
// ignored.  In GraphBLAS notation, C{A} = A'*A followed by reduce(C) to scalar.
// This method is for testing only, to check the result of other, faster
// methods.  Do not benchmark this method; it is slow and simple by design.

#define LAGraph_FREE_ALL                        \
{                                               \
    LAGraph_Free ((void **) &Ap) ;              \
    LAGraph_Free ((void **) &Aj) ;              \
    LAGraph_Free ((void **) &Ax) ;              \
}

#include "LG_internal.h"
#include "LG_test.h"

//------------------------------------------------------------------------------
// LG_check_tri
//------------------------------------------------------------------------------

// Since this method does not modify G->A, it can be tested with LG_BRUTAL.
// See test_TriangleCount for a brutal memory test of this method.

int LG_check_tri        // -1 if out of memory, 0 if successful
(
    // output
    uint64_t *ntri,     // # of triangles in A
    // input
    LAGraph_Graph G,    // the structure of G->A must be symmetric
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;

    GrB_Index *Ap = NULL, *Aj = NULL, *Ai = NULL ;
    void *Ax = NULL ;
    GrB_Index Ap_size, Aj_size, Ax_size, n, ncols, Ap_len, Aj_len, Ax_len ;
    LG_CHECK (ntri == NULL, -1003, "ntri is NULL") ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), -1002, "graph is invalid") ;
    LG_CHECK (G->ndiag != 0, -104, "G->ndiag must be zero") ;
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
        LG_CHECK (false, -1005, "G->A must be symmetric") ;
    }
    GrB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GrB_TRY (GrB_Matrix_ncols (&ncols, G->A)) ;
    LG_CHECK (n != ncols, -1001, "A must be square") ;

    //--------------------------------------------------------------------------
    // export the matrix in CSR form
    //--------------------------------------------------------------------------

    size_t typesize ;
    LAGraph_TRY (LG_check_export (G, &Ap, &Aj, &Ax, &Ap_len, &Aj_len, &Ax_len,
        &typesize, msg)) ;

    //--------------------------------------------------------------------------
    // compute the # of triangles (each triangle counted 6 times)
    //--------------------------------------------------------------------------

    int64_t ntriangles = 0 ;
    Ai = Aj ;       // pretend A is symmetric and in CSC format instead

    // masked dot-product method
    #if !defined ( COVERAGE )
    #pragma omp parallel for reduction(+:ntriangles) schedule(dynamic,1024)
    #endif
    for (int64_t j = 0 ; j < n ; j++)
    {
        // for each entry in the lower triangular part of A
        for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            const int64_t i = Ai [p] ;
            if (i > j)
            {
                // ntriangles += A(:,i)' * A(:,j)
                int64_t p1 = Ap [i] ;
                int64_t p1_end = Ap [i+1] ;
                int64_t p2 = Ap [j] ;
                int64_t p2_end = Ap [j+1] ;
                while (p1 < p1_end && p2 < p2_end)
                {
                    int64_t i1 = Ai [p1] ;
                    int64_t i2 = Ai [p2] ;
                    if (i1 < i2)
                    { 
                        // A(i1,i) appears before A(i2,j)
                        p1++ ;
                    }
                    else if (i2 < i1)
                    { 
                        // A(i2,j) appears before A(i1,i)
                        p2++ ;
                    }
                    else // i1 == i2 == k
                    { 
                        // A(k,i) and A(k,j) are the next entries to merge
                        ntriangles++ ;
                        p1++ ;
                        p2++ ;
                    }
                }
            }
        }
    }
    ntriangles = ntriangles / 3 ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LAGraph_FREE_ALL ;
    (*ntri) = ntriangles ;
    return (0) ;
}

