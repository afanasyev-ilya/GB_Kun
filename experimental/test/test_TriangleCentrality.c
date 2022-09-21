//----------------------------------------------------------------------------
// LAGraph/src/test/test_TriangleCentrality.cpp: test cases for triangle
// centrality
// ----------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix C = NULL ;
GrB_Type atype = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    uint64_t ntriangles ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    {     11, "A.mtx" },
    {   2016, "jagmesh7.mtx" },
    { 342300, "bcsstk13.mtx" },
    {     45, "karate.mtx" },
    {      6, "ldbc-cdlp-undirected-example.mtx" },
    {      4, "ldbc-undirected-example-bool.mtx" },
    {      4, "ldbc-undirected-example-unweighted.mtx" },
    {      4, "ldbc-undirected-example.mtx" },
    {      5, "ldbc-wcc-example.mtx" },
    { 0, "" },
} ;

//****************************************************************************
void test_TriangleCentrality (void)
{
    LAGraph_Init (msg) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        uint64_t ntriangles = files [k].ntriangles ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, &atype, f, msg)) ;

        // C = spones (A), in FP64, required for methods 1 and 1.5
        GrB_Index n ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        OK (GrB_Matrix_new (&C, GrB_FP64, n, n)) ;
        OK (GrB_assign (C, A, NULL, (double) 1, GrB_ALL, n, GrB_ALL, n,
            GrB_DESC_S)) ;
        OK (GrB_free (&A)) ;
        TEST_CHECK (A == NULL) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct an undirected graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, atype, LAGRAPH_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

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

        uint64_t ntri ;
        GrB_Vector c = NULL ;
        for (int method = 0 ; method <= 3 ; method++)
        {
            printf ("\nMethod: %d\n", method) ;

            // compute the triangle centrality
            OK (LAGraph_VertexCentrality_Triangle (&c, &ntri, method, G, msg)) ;
            printf ("# of triangles: %g\n", (double) ntri) ;
            TEST_CHECK (ntri == ntriangles) ;

            int pr = (n <= 100) ? 3 : 2 ;
            printf ("\ncentrality:\n") ;
            OK (LAGraph_Vector_print (c, pr, stdout, msg)) ;
            OK (GrB_free (&c)) ;
        }

        // convert to directed with symmetric structure and recompute
        G->kind = LAGRAPH_ADJACENCY_DIRECTED ;
        OK (LAGraph_VertexCentrality_Triangle (&c, &ntri, 0, G, msg)) ;
        TEST_CHECK (ntri == ntriangles) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_errors
//------------------------------------------------------------------------------

void test_errors (void)
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

    uint64_t ntri ;
    GrB_Vector c = NULL ;

    // c is NULL
    int result = LAGraph_VertexCentrality_Triangle (NULL, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G is invalid
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, NULL, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_INVALID_OBJECT) ;
    TEST_CHECK (c == NULL) ;

    // G may have self edges
    G->ndiag = LAGRAPH_UNKNOWN ;
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1004) ;
    TEST_CHECK (c == NULL) ;

    // G is undirected
    G->ndiag = 0 ;
    G->kind = LAGRAPH_ADJACENCY_DIRECTED ;
    G->A_structure_is_symmetric = LAGRAPH_FALSE ;
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1005) ;
    TEST_CHECK (c == NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}


//****************************************************************************

TEST_LIST = {
    {"TriangleCentrality", test_TriangleCentrality},
    {"TriangleCentrality_errors", test_errors},
    {NULL, NULL}
};
