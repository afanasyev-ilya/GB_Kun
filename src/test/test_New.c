//------------------------------------------------------------------------------
// LAGraph/src/test/test_New.c:  test LAGraph_New and LAGraph_Delete
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

LAGraph_Graph G = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL ;
GrB_Type atype = NULL ;
#define LEN 512
char filename [LEN+1] ;

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
// test_New:  test LAGraph_New
//------------------------------------------------------------------------------

typedef struct
{
    LAGraph_Kind kind ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    LAGRAPH_ADJACENCY_DIRECTED,   "cover.mtx",
    LAGRAPH_ADJACENCY_DIRECTED,   "ldbc-directed-example.mtx",
    LAGRAPH_ADJACENCY_UNDIRECTED, "ldbc-undirected-example.mtx",
    LAGRAPH_UNKNOWN,              ""
} ;

void test_New (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        OK (LAGraph_New (&G, &A, atype, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // check the graph
        OK (LAGraph_CheckGraph (G, msg)) ;
        TEST_CHECK (G->kind == kind) ;
        if (kind == LAGRAPH_ADJACENCY_DIRECTED)
        {
            TEST_CHECK (G->A_structure_is_symmetric == LAGRAPH_UNKNOWN) ;
        }
        else
        {
            TEST_CHECK (G->A_structure_is_symmetric == LAGRAPH_TRUE) ;
        }

        // free the graph
        OK (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;
    }
    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_New_brutal
//------------------------------------------------------------------------------

#if LG_SUITESPARSE
void test_New_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, &atype, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        LG_BRUTAL_BURBLE (LAGraph_New (&G, &A, atype, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // check the graph
        LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;

        // free the graph
        LG_BRUTAL_BURBLE (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// test_New_failures:  test error handling of LAGraph_New
//------------------------------------------------------------------------------

void test_New_failures (void)
{
    setup ( ) ;

    // G cannot be NULL
    TEST_CHECK (LAGraph_New (NULL, NULL, NULL, 0, msg) == -1) ;
    printf ("\nmsg: %s\n", msg) ;

    // create a graph with no adjacency matrix; this is OK, since the intent is
    // to create a graph for which the adjacency matrix can be defined later,
    // via assigning it to G->A.  However, the graph will be declared invalid
    // by LAGraph_CheckGraph since G->A is NULL.
    OK (LAGraph_New (&G, NULL, NULL, 0, msg)) ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == -1102) ;
    printf ("msg: %s\n", msg) ;
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;
    OK (LAGraph_Delete (NULL, msg)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "New", test_New },
    { "New_failures", test_New_failures },
    { "New_brutal", test_New_brutal },
    { NULL, NULL }
} ;

