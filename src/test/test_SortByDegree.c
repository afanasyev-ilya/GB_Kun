//------------------------------------------------------------------------------
// LAGraph/src/test/test_SortByDegree  test LAGraph_SortByDegree
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

#include "LAGraph_test.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

LAGraph_Graph G = NULL, H = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B = NULL ;
GrB_Vector d = NULL ;
GrB_Type atype = NULL ;
#define LEN 512
char filename [LEN+1] ;
int64_t *P = NULL ;
bool *W = NULL ;
GrB_Index n, nrows, ncols ;
bool is_symmetric ;
int kind ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_SortByDegree  test LAGraph_SortByDegree
//------------------------------------------------------------------------------

const char *files [ ] =
{
    "A.mtx", 
    "LFAT5.mtx", 
    "cover.mtx", 
    "full.mtx", 
    "full_symmetric.mtx", 
    "karate.mtx", 
    "ldbc-cdlp-directed-example.mtx", 
    "ldbc-cdlp-undirected-example.mtx", 
    "ldbc-directed-example-bool.mtx", 
    "ldbc-directed-example-unweighted.mtx", 
    "ldbc-directed-example.mtx", 
    "ldbc-undirected-example-bool.mtx", 
    "ldbc-undirected-example-unweighted.mtx", 
    "ldbc-undirected-example.mtx", 
    "ldbc-wcc-example.mtx", 
    "matrix_int16.mtx", 
    "msf1.mtx", 
    "msf2.mtx", 
    "msf3.mtx", 
    "structure.mtx", 
    "sample.mtx", 
    "sample2.mtx", 
    "skew_fp32.mtx", 
    "tree-example.mtx", 
    "west0067.mtx", 
    "",
} ;

void test_SortByDegree (void)
{
    setup ( ) ;

    for (int kk = 0 ; ; kk++)
    {

        // get the name of the test matrix
        const char *aname = files [kk] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n############################################# %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;

        for (int outer = 0 ; outer <= 1 ; outer++)
        {

            // load the matrix as A
            FILE *f = fopen (filename, "r") ;
            TEST_CHECK (f != NULL) ;
            OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
            OK (fclose (f)) ;
            TEST_MSG ("Loading of adjacency matrix failed") ;

            // ensure the matrix is square
            OK (GrB_Matrix_nrows (&nrows, A)) ;
            OK (GrB_Matrix_ncols (&ncols, A)) ;
            TEST_CHECK (nrows == ncols) ;
            n = nrows ;

            // decide if the graph G is directed or undirected
            if (outer == 0)
            {
                kind = LAGRAPH_ADJACENCY_DIRECTED ;
                printf ("\n#### case: directed graph\n\n") ;
            }
            else
            {
                kind = LAGRAPH_ADJACENCY_UNDIRECTED ;
                printf ("\n#### case: undirected graph\n\n") ;
            }

            // construct a graph G with adjacency matrix A
            TEST_CHECK (A != NULL) ;
            OK (LAGraph_New (&G, &A, atype, kind, msg)) ;
            TEST_CHECK (A == NULL) ;
            TEST_CHECK (G != NULL) ;

            // create the properties
            OK (LAGraph_Property_AT (G, msg)) ;
            OK (LAGraph_Property_RowDegree (G, msg)) ;
            OK (LAGraph_Property_ColDegree (G, msg)) ;
            OK (LAGraph_Property_ASymmetricStructure (G, msg)) ;
            OK (LAGraph_DisplayGraph (G, 2, stdout, msg)) ;

            // sort 4 different ways
            for (int trial = 0 ; trial <= 3 ; trial++)
            {
                bool byrow = (trial == 0 || trial == 1) ;
                bool ascending = (trial == 0 || trial == 2) ;

                // sort the graph by degree
                TEST_CHECK (P == NULL) ;
                OK (LAGraph_SortByDegree (&P, G, byrow, ascending, msg)) ;
                TEST_CHECK (P != NULL) ;

                // ensure P is a permutation of 0..n-1
                W = (bool *) LAGraph_Calloc (n, sizeof (bool)) ;
                TEST_CHECK (W != NULL) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t j = P [k] ;
                    TEST_CHECK (j >= 0 && j < n) ;
                    TEST_CHECK (W [j] == false) ;
                    W [j] = true ;
                }

                // check the result by constructing a new graph with adjacency
                // matrix B = A (P,P)
                OK (GrB_Matrix_new (&B, GrB_BOOL, n, n)) ;
                OK (GrB_extract (B, NULL, NULL, G->A,
                    (GrB_Index *) P, n, (GrB_Index *) P, n, NULL)) ;
                OK (LAGraph_New (&H, &B, GrB_BOOL, kind, msg)) ;
                TEST_CHECK (B == NULL) ;
                TEST_CHECK (H != NULL) ;

                // get the properties of H
                OK (LAGraph_Property_RowDegree (H, msg)) ;
                OK (LAGraph_Property_ColDegree (H, msg)) ;
                OK (LAGraph_Property_ASymmetricStructure (H, msg)) ;
                TEST_CHECK (G->A_structure_is_symmetric ==
                            H->A_structure_is_symmetric) ;
                printf ("\nTrial %d, graph H, sorted (%s) by (%s) degrees:\n",
                    trial, ascending ? "ascending" : "descending",
                    byrow ? "row" : "column") ;
                OK (LAGraph_DisplayGraph (H, 2, stdout, msg)) ;

                d = (byrow || G->A_structure_is_symmetric) ?
                    H->rowdegree : H->coldegree ;

                // ensure d is sorted in ascending or descending order
                int64_t last_deg = (ascending) ? (-1) : (n+1) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t deg = 0 ;
                    GrB_Info info = GrB_Vector_extractElement (&deg, d, k) ;
                    TEST_CHECK (info == GrB_NO_VALUE || info == GrB_SUCCESS) ;
                    if (info == GrB_NO_VALUE) deg = 0 ;
                    if (ascending)
                    {
                        TEST_CHECK (last_deg <= deg) ;
                    }
                    else
                    {
                        TEST_CHECK (last_deg >= deg) ;
                    }
                    last_deg = deg ;
                }

                // free workspace and the graph H
                LAGraph_Free ((void **) &W) ;
                LAGraph_Free ((void **) &P) ;
                OK (LAGraph_Delete (&H, msg)) ;
            }

            // check if the adjacency matrix is symmetric
            if (outer == 0)
            {
                // if G->A is symmetric, then continue the outer iteration to
                // create an undirected graph.  Otherwise just do the directed
                // graph
                OK (LAGraph_IsEqual_type (&is_symmetric, G->A, G->AT,
                    atype, msg)) ;
                if (!is_symmetric)
                {
                    printf ("matrix is unsymmetric; skip undirected case\n") ;
                    OK (LAGraph_Delete (&G, msg)) ;
                    break ;
                }
            }
            OK (LAGraph_Delete (&G, msg)) ;
        }
    }

    teardown ( ) ;
}


//------------------------------------------------------------------------------
// test_SortByDegree_brutal
//------------------------------------------------------------------------------

#if LG_SUITESPARSE
void test_SortByDegree_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

    for (int kk = 0 ; ; kk++)
    {

        // get the name of the test matrix
        const char *aname = files [kk] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("%s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;

        for (int outer = 0 ; outer <= 1 ; outer++)
        {

            // load the matrix as A
            FILE *f = fopen (filename, "r") ;
            TEST_CHECK (f != NULL) ;
            OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
            OK (fclose (f)) ;
            TEST_MSG ("Loading of adjacency matrix failed") ;

            // ensure the matrix is square
            OK (GrB_Matrix_nrows (&nrows, A)) ;
            OK (GrB_Matrix_ncols (&ncols, A)) ;
            TEST_CHECK (nrows == ncols) ;
            n = nrows ;

            // decide if the graph G is directed or undirected
            if (outer == 0)
            {
                kind = LAGRAPH_ADJACENCY_DIRECTED ;
            }
            else
            {
                kind = LAGRAPH_ADJACENCY_UNDIRECTED ;
            }

            // construct a graph G with adjacency matrix A
            TEST_CHECK (A != NULL) ;
            OK (LAGraph_New (&G, &A, atype, kind, msg)) ;
            TEST_CHECK (A == NULL) ;
            TEST_CHECK (G != NULL) ;

            // create the properties
            OK (LAGraph_Property_AT (G, msg)) ;
            OK (LAGraph_Property_RowDegree (G, msg)) ;
            OK (LAGraph_Property_ColDegree (G, msg)) ;
            OK (LAGraph_Property_ASymmetricStructure (G, msg)) ;
            // OK (LAGraph_DisplayGraph (G, 2, stdout, msg)) ;

            // sort 4 different ways
            for (int trial = 0 ; trial <= 3 ; trial++)
            {
                bool byrow = (trial == 0 || trial == 1) ;
                bool ascending = (trial == 0 || trial == 2) ;

                // sort the graph by degree
                TEST_CHECK (P == NULL) ;
                OK (LAGraph_SortByDegree (&P, G, byrow, ascending,
                    msg)) ;
                TEST_CHECK (P != NULL) ;

                // ensure P is a permutation of 0..n-1
                W = (bool *) LAGraph_Calloc (n, sizeof (bool)) ;
                TEST_CHECK (W != NULL) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t j = P [k] ;
                    TEST_CHECK (j >= 0 && j < n) ;
                    TEST_CHECK (W [j] == false) ;
                    W [j] = true ;
                }

                // check the result by constructing a new graph with adjacency
                // matrix B = A (P,P)
                OK (GrB_Matrix_new (&B, GrB_BOOL, n, n)) ;
                OK (GrB_extract (B, NULL, NULL, G->A,
                    (GrB_Index *) P, n, (GrB_Index *) P, n, NULL)) ;
                OK (LAGraph_New (&H, &B, GrB_BOOL, kind, msg)) ;
                TEST_CHECK (B == NULL) ;
                TEST_CHECK (H != NULL) ;

                // get the properties of H
                OK (LAGraph_Property_RowDegree (H, msg)) ;
                OK (LAGraph_Property_ColDegree (H, msg)) ;
                OK (LAGraph_Property_ASymmetricStructure (H, msg)) ;
                TEST_CHECK (G->A_structure_is_symmetric ==
                            H->A_structure_is_symmetric) ;

                d = (byrow || G->A_structure_is_symmetric) ?
                    H->rowdegree : H->coldegree ;

                // ensure d is sorted in ascending or descending order
                int64_t last_deg = (ascending) ? (-1) : (n+1) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t deg = 0 ;
                    GrB_Info info = GrB_Vector_extractElement (&deg, d, k) ;
                    TEST_CHECK (info == GrB_NO_VALUE || info == GrB_SUCCESS) ;
                    if (info == GrB_NO_VALUE) deg = 0 ;
                    if (ascending)
                    {
                        TEST_CHECK (last_deg <= deg) ;
                    }
                    else
                    {
                        TEST_CHECK (last_deg >= deg) ;
                    }
                    last_deg = deg ;
                }

                // free workspace and the graph H
                LAGraph_Free ((void **) &W) ;
                LAGraph_Free ((void **) &P) ;
                OK (LAGraph_Delete (&H, msg)) ;
            }

            // check if the adjacency matrix is symmetric
            if (outer == 0)
            {
                // if G->A is symmetric, then continue the outer iteration to
                // create an undirected graph.  Otherwise just do the directed
                // graph
                OK (LAGraph_IsEqual_type (&is_symmetric, G->A, G->AT,
                    atype, msg)) ;
                if (!is_symmetric)
                {
                    OK (LAGraph_Delete (&G, msg)) ;
                    break ;
                }
            }

            OK (LAGraph_Delete (&G, msg)) ;
        }
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// test_SortByDegree_failures:  test error handling of LAGraph_SortByDegree
//-----------------------------------------------------------------------------

void test_SortByDegree_failures (void)
{
    setup ( ) ;

    TEST_CHECK (LAGraph_SortByDegree (NULL, NULL, true, true, msg) == -1) ;
    printf ("\nmsg: %s\n", msg) ;

    TEST_CHECK (LAGraph_SortByDegree (&P, NULL, true, true, msg) == -1) ;
    printf ("msg: %s\n", msg) ;

    // create the karate graph
    FILE *f = fopen (LG_DATA_DIR "karate.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;
    OK (LAGraph_New (&G, &A, atype, LAGRAPH_ADJACENCY_UNDIRECTED, msg)) ;

    // degree property must first be computed
    TEST_CHECK (LAGraph_SortByDegree (&P, G, true, true, msg) == -1) ;
    printf ("msg: %s\n", msg) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "SortByDegree", test_SortByDegree },
    { "SortByDegree_failures", test_SortByDegree_failures },
    #if LG_SUITESPARSE
    { "SortByDegree_brutal", test_SortByDegree_brutal },
    #endif
    { NULL, NULL }
} ;

