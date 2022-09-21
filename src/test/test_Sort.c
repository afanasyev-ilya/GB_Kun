//-----------------------------------------------------------------------------
// LAGraph/src/test/test_Sort.c: test LAGraph_Sort* methods
//-----------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//-----------------------------------------------------------------------------

#include "LAGraph_test.h"
#include "LG_internal.h"

char msg [LAGRAPH_MSG_LEN] ;

//-----------------------------------------------------------------------------
// test_sort1
//-----------------------------------------------------------------------------

void test_sort1 (void)
{
    OK (LAGraph_Init (msg)) ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        int64_t n = (trial == 0) ? 1024 : (256 * 1024) ;

        int64_t *A0 = LAGraph_Malloc (n, sizeof (int64_t)) ;

        TEST_CHECK (A0 != NULL) ;

        uint64_t seed = 1 ;
        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) LAGraph_Random15 (&seed) ;
        }

        OK (LAGraph_Sort1 (A0, n, 4, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) (LAGraph_Random15 (&seed) % 4) ;
        }

        OK (LAGraph_Sort1 (A0, n, 4, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        LAGraph_Free ((void **) &A0) ;
    }

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_sort2
//-----------------------------------------------------------------------------

void test_sort2 (void)
{
    OK (LAGraph_Init (msg)) ;

    int64_t n = 256 * 1024 ;

    int64_t *A0 = LAGraph_Malloc (n, sizeof (int64_t)) ;
    int64_t *A1 = LAGraph_Malloc (n, sizeof (int64_t)) ;

    TEST_CHECK (A0 != NULL) ;
    TEST_CHECK (A1 != NULL) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LAGraph_Random15 (&seed) ;
        A1 [k] = (int64_t) LAGraph_Random60 (&seed) ;
    }

    OK (LAGraph_Sort2 (A0, A1, n, 4, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LAGraph_Random15 (&seed) % 4) ;
    }

    OK (LAGraph_Sort2 (A0, A1, n, 4, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    LAGraph_Free ((void **) &A0) ;
    LAGraph_Free ((void **) &A1) ;

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// test_sort1_brutal
//-----------------------------------------------------------------------------

#if LG_SUITESPARSE
void test_sort1_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        int64_t n = (trial == 0) ? 1024 : (256 * 1024) ;

        int64_t *A0 = LAGraph_Malloc (n, sizeof (int64_t)) ;

        TEST_CHECK (A0 != NULL) ;

        uint64_t seed = 1 ;
        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) LAGraph_Random15 (&seed) ;
        }

        LG_BRUTAL (LAGraph_Sort1 (A0, n, 4, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        for (int k = 0 ; k < n ; k++)
        {
            A0 [k] = (int64_t) (LAGraph_Random15 (&seed) % 4) ;
        }

        LG_BRUTAL (LAGraph_Sort1 (A0, n, 4, msg)) ;

        for (int k = 1 ; k < n ; k++)
        {
            TEST_CHECK (A0 [k-1] <= A0 [k]) ;
        }

        LAGraph_Free ((void **) &A0) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// test_sort2_brutal
//-----------------------------------------------------------------------------

#if LG_SUITESPARSE
void test_sort2_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    int64_t n = 256 * 1024 ;

    int64_t *A0 = LAGraph_Malloc (n, sizeof (int64_t)) ;
    int64_t *A1 = LAGraph_Malloc (n, sizeof (int64_t)) ;

    TEST_CHECK (A0 != NULL) ;
    TEST_CHECK (A1 != NULL) ;

    uint64_t seed = 1 ;
    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = (int64_t) LAGraph_Random15 (&seed) ;
        A1 [k] = (int64_t) LAGraph_Random60 (&seed) ;
    }

    LG_BRUTAL (LAGraph_Sort2 (A0, A1, n, 4, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    for (int k = 0 ; k < n ; k++)
    {
        A0 [k] = 0 ;
        A1 [k] = (int64_t) (LAGraph_Random15 (&seed) % 4) ;
    }

    LG_BRUTAL (LAGraph_Sort2 (A0, A1, n, 4, msg)) ;

    for (int k = 1 ; k < n ; k++)
    {
        TEST_CHECK (LG_lt_2 (A0, A1, k-1, A0, A1, k)
            || (A0 [k-1] == A0 [k] && A1 [k-1] == A1 [k])) ;
    }

    LAGraph_Free ((void **) &A0) ;
    LAGraph_Free ((void **) &A1) ;

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST = {
    {"test_sort1", test_sort1},
    {"test_sort2", test_sort2},
    #if LG_SUITESPARSE
    {"test_sort1_brutal", test_sort1_brutal},
    {"test_sort2_brutal", test_sort2_brutal},
    #endif
    {NULL, NULL}
};
