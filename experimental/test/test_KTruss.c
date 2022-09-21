//----------------------------------------------------------------------------
// LAGraph/experimental/test/test_KTruss.c: test cases for k-truss
// ----------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include "LAGraphX.h"
#include "LAGraph_test.h"
#include "LG_Xtest.h"

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix C1 = NULL, C2 = NULL ;
GrB_Type atype = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    uint32_t ntriangles ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    {     11, "A.mtx" },
    {   2016, "jagmesh7.mtx" },
//  { 342300, "bcsstk13.mtx" },
    {     45, "karate.mtx" },
    {      6, "ldbc-cdlp-undirected-example.mtx" },
    {      4, "ldbc-undirected-example-bool.mtx" },
    {      4, "ldbc-undirected-example-unweighted.mtx" },
    {      4, "ldbc-undirected-example.mtx" },
    {      5, "ldbc-wcc-example.mtx" },
    { 0, "" },
} ;

//****************************************************************************
void test_ktruss (void)
{
    LAGraph_Init (msg) ;

    for (int id = 0 ; ; id++)
    {

        // load the matrix as A
        const char *aname = files [id].name ;
        uint32_t ntriangles = files [id].ntriangles ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct an undirected graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, atype, LAGRAPH_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // check for self-edges
        OK (LAGraph_Property_NDiag (G, msg)) ;
        if (G->ndiag != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->ndiag) ;
            OK (LAGraph_DeleteDiag (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->ndiag) ;
            TEST_CHECK (G->ndiag == 0) ;
        }

        // compute each k-truss until the result is empty
        bool ok ;
        GrB_Index n ;
        OK (GrB_Matrix_nrows (&n, G->A)) ;
        for (int k = 3 ; k < n ; k++)
        {
            // compute the k-truss
            printf ("\n%d-truss:\n", k) ;
            OK (LAGraph_KTruss (&C1, G, k, msg)) ;

            // compute it again to check the result
            OK (LG_check_ktruss (&C2, G, k, msg)) ;
            OK (LAGraph_IsEqual (&ok, C1, C2, msg)) ;
            TEST_CHECK (ok) ;

            // count the triangles in the 3-truss
            if (k == 3)
            {
                uint32_t nt = 0 ;
                OK (GrB_reduce (&nt, NULL, GrB_PLUS_MONOID_UINT32, C1, NULL)) ;
                nt = nt / 6 ;
                TEST_CHECK (nt == ntriangles) ;
            }

            // free C1 and C2, and break if C1 is empty
            GrB_Index nvals ;
            OK (GrB_Matrix_nvals (&nvals, C1)) ;
            OK (GrB_free (&C1)) ;
            OK (GrB_free (&C2)) ;
            if (nvals == 0) break ;
        }

        // convert to directed with symmetric structure and recompute
        G->kind = LAGRAPH_ADJACENCY_DIRECTED ;
        G->A_structure_is_symmetric = true ;
        OK (LAGraph_KTruss (&C1, G, 3, msg)) ;
        OK (LG_check_ktruss (&C2, G, 3, msg)) ;
        OK (LAGraph_IsEqual (&ok, C1, C2, msg)) ;
        TEST_CHECK (ok) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_ktruss_error
//------------------------------------------------------------------------------

void test_ktruss_errors (void)
{
    LAGraph_Init (msg) ;

    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;

    // construct an undirected graph G with adjacency matrix A
    OK (LAGraph_New (&G, &A, atype, LAGRAPH_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;

    OK (LAGraph_Property_NDiag (G, msg)) ;

    // C is NULL
    int result = LAGraph_KTruss (NULL, G, 3, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // k is invalid
    result = LAGraph_KTruss (&C1, G, 2, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_INVALID_VALUE) ;
    TEST_CHECK (C1 == NULL) ;

    // G is invalid
    result = LAGraph_KTruss (&C1, NULL, 3, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_INVALID_OBJECT) ;
    TEST_CHECK (C1 == NULL) ;

    // G may have self edges
    G->ndiag = LAGRAPH_UNKNOWN ;
    result = LAGraph_KTruss (&C1, G, 3, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1004) ;
    TEST_CHECK (C1 == NULL) ;

    // G is undirected
    G->ndiag = 0 ;
    G->kind = LAGRAPH_ADJACENCY_DIRECTED ;
    G->A_structure_is_symmetric = LAGRAPH_FALSE ;
    result = LAGraph_KTruss (&C1, G, 3, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1005) ;
    TEST_CHECK (C1 == NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"ktruss", test_ktruss},
    {"ktruss_errors", test_ktruss_errors},
    {NULL, NULL}
};
