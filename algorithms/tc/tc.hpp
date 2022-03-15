// code taken form LAGraph
// add license

#pragma once
#include "../../src/gb_kun.h"

#define GrB_Matrix lablas::Matrix<int>*
#define GrB_Vector lablas::Vector<int>*
#define MASK_NULL static_cast<const lablas::Vector<int>*>(NULL)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum LAGraph_TriangleCount_Method
{
    LAGraph_TriangleCount_Default = 0,
    LAGraph_TriangleCount_Burkhardt = 1,
    LAGraph_TriangleCount_Cohen = 2,
    LAGraph_TriangleCount_Sandia = 3,
    LAGraph_TriangleCount_Sandia2 = 4,
    LAGraph_TriangleCount_SandiaDot = 5,
    LAGraph_TriangleCount_SandiaDot2 = 6
};

enum LAGraph_TriangleCount_Presort
{
    LAGraph_TriangleCount_NoSort,
    LAGraph_TriangleCount_Ascending,
    LAGraph_TriangleCount_Descending,
    LAGraph_TriangleCount_AutoSort
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
#define LG_FREE_ALL             \
{                               \
    GrB_free (L) ;              \
    GrB_free (U) ;              \
}

static int tricount_prep
        (
                GrB_Matrix *L,      // if present, compute L = tril (A,-1)
                GrB_Matrix *U,      // if present, compute U = triu (A, 1)
                GrB_Matrix A,       // input matrix
                char *msg
        )
{
    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, A)) ;

    if (L != NULL)
    {
        // L = tril (A,-1)
        GrB_TRY (GrB_Matrix_new (L, GrB_BOOL, n, n)) ;
        GrB_TRY (GrB_select (*L, NULL, NULL, GrB_TRIL, A, (int64_t) (-1),
                             NULL)) ;
    }

    if (U != NULL)
    {
        // U = triu (A,1)
        GrB_TRY (GrB_Matrix_new (U, GrB_BOOL, n, n)) ;
        GrB_TRY (GrB_select (*U, NULL, NULL, GrB_TRIU, A, (int64_t) 1, NULL)) ;
    }
    return (GrB_SUCCESS) ;
}
#undef  LG_FREE_ALL
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
#define LG_FREE_ALL                         \
{                                           \
    GrB_free (&C) ;                         \
    GrB_free (&L) ;                         \
    GrB_free (&T) ;                         \
    GrB_free (&U) ;                         \
    LAGraph_Free ((void **) &P) ;           \
}
*/
int LAGr_TriangleCount(uint64_t *ntriangles, const LAGraph_Graph<int>* G,
                       LAGraph_TriangleCount_Method method,
                       LAGraph_TriangleCount_Presort presort, const char *msg)
{
    LG_CLEAR_MSG ;
    GrB_Matrix C = NULL, L = NULL, U = NULL, T = NULL ;
    int64_t *P = NULL ;
    /*
    LG_ASSERT_MSG (
            method == LAGraph_TriangleCount_Default ||   // 0: use default method
            method == LAGraph_TriangleCount_Burkhardt || // 1: sum (sum ((A^2) .* A))/6
            method == LAGraph_TriangleCount_Cohen ||     // 2: sum (sum ((L * U) .*A))/2
            method == LAGraph_TriangleCount_Sandia ||    // 3: sum (sum ((L * L) .* L))
            method == LAGraph_TriangleCount_Sandia2 ||   // 4: sum (sum ((U * U) .* U))
            method == LAGraph_TriangleCount_SandiaDot || // 5: sum (sum ((L * U') .* L))
            method == LAGraph_TriangleCount_SandiaDot2,  // 6: sum (sum ((U * L') .* U))
            GrB_INVALID_VALUE, "method is invalid") ;
    if (presort != NULL)
    {
        LG_ASSERT_MSG (
                (*presort) == LAGraph_TriangleCount_NoSort ||
                (*presort) == LAGraph_TriangleCount_Ascending ||
                (*presort) == LAGraph_TriangleCount_Descending ||
                (*presort) == LAGraph_TriangleCount_AutoSort,
                GrB_INVALID_VALUE, "presort is invalid") ;
    }
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    LG_ASSERT (ntriangles != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (G->ndiag == 0, LAGRAPH_NO_SELF_EDGES_ALLOWED) ;

    if (method == LAGraph_TriangleCount_Default)
    {
        // 0: use default method (5): SandiaDot: sum (sum ((L * U') .* L))
        method = LAGraph_TriangleCount_SandiaDot ;
    }

    LG_ASSERT_MSG ((G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
                    (G->kind == LAGraph_ADJACENCY_DIRECTED &&
                     G->structure_is_symmetric == LAGraph_TRUE)),
                   LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED,
                   "G->A must be known to be symmetric") ;

    // the Sandia* methods can benefit from the presort
    bool method_can_use_presort =
            method == LAGraph_TriangleCount_Sandia ||    // 3: sum (sum ((L * L) .* L))
            method == LAGraph_TriangleCount_Sandia2 ||   // 4: sum (sum ((U * U) .* U))
            method == LAGraph_TriangleCount_SandiaDot || // 5: sum (sum ((L * U') .* L))
            method == LAGraph_TriangleCount_SandiaDot2 ; // 6: sum (sum ((U * L') .* U))
    */
    GrB_Matrix A = G->A ;
    auto Degree = G->rowdegree ;
    /*
    bool auto_sort = (presort != NULL)
                     && ((*presort) == LAGraph_TriangleCount_AutoSort) ;
    if (auto_sort && method_can_use_presort)
    {
        LG_ASSERT_MSG (Degree != NULL,
                       LAGRAPH_PROPERTY_MISSING, "G->rowdegree is required") ;
    }
    */
    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GrB_TRY (GrB_Matrix_new (&C, GrB_INT64, n, n)) ;
    auto semiring = LAGraph_plus_one_int64 ;
    auto monoid = GrB_PLUS_MONOID_INT64 ;
    /*
    if (auto_sort)
    {
        // auto selection of sorting method
        (*presort) = LAGraph_TriangleCount_NoSort ; // default is not to sort

        if (method_can_use_presort)
        {
            #define NSAMPLES 1000
            GrB_Index nvals ;
            GrB_TRY (GrB_Matrix_nvals (&nvals, A)) ;
            if (n > NSAMPLES && ((double) nvals / ((double) n)) >= 10)
            {
                // estimate the mean and median degrees
                double mean, median ;
                LG_TRY (LAGraph_SampleDegree (&mean, &median,
                                              G, true, NSAMPLES, n, msg)) ;
                // sort if the average degree is very high vs the median
                if (mean > 4 * median)
                {
                    switch (method)
                    {
                        case LAGraph_TriangleCount_Sandia:
                            // 3:sum (sum ((L * L) .* L))
                            (*presort) = LAGraph_TriangleCount_Ascending  ;
                            break ;
                        case LAGraph_TriangleCount_Sandia2:
                            // 4: sum (sum ((U * U) .* U))
                            (*presort) = LAGraph_TriangleCount_Descending ;
                            break ;
                        default:
                        case LAGraph_TriangleCount_SandiaDot:
                            // 5: sum (sum ((L * U') .* L))
                            (*presort) = LAGraph_TriangleCount_Ascending  ;
                            break ;
                        case LAGraph_TriangleCount_SandiaDot2:
                            // 6: sum (sum ((U * L') .* U))
                            (*presort) = LAGraph_TriangleCount_Descending ;
                            break ;
                    }
                }
            }
        }
    }

    if (presort != NULL && (*presort) != LAGraph_TriangleCount_NoSort)
    {
        // P = permutation that sorts the rows by their degree
        LG_TRY (LAGraph_SortByDegree (&P, G, true,
                                      (*presort) == LAGraph_TriangleCount_Ascending, msg)) ;

        // T = A (P,P) and typecast to boolean
        GrB_TRY (GrB_Matrix_new (&T, GrB_BOOL, n, n)) ;
        GrB_TRY (GrB_extract (T, NULL, NULL, A, (GrB_Index *) P, n,
                              (GrB_Index *) P, n, NULL)) ;
        A = T ;

        // free workspace
        LAGraph_Free ((void **) &P) ;
    }
    */
    int64_t ntri ;

    switch (method)
    {

        case LAGraph_TriangleCount_Burkhardt:  // 1: sum (sum ((A^2) .* A)) / 6

            GrB_TRY (GrB_mxm (C, A, NULL, semiring, A, A, GrB_DESC_S)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_Cohen: // 2: sum (sum ((L * U) .* A)) / 2
            /*
            LG_TRY (tricount_prep (&L, &U, A, msg)) ;
            GrB_TRY (GrB_mxm (C, A, NULL, semiring, L, U, GrB_DESC_S)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 2 ;
            */
            break ;

        case LAGraph_TriangleCount_Sandia: // 3: sum (sum ((L * L) .* L))
            /*
            // using the masked saxpy3 method
            LG_TRY (tricount_prep (&L, NULL, A, msg)) ;
            GrB_TRY (GrB_mxm (C, L, NULL, semiring, L, L, GrB_DESC_S)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
             */
            break ;

        case LAGraph_TriangleCount_Sandia2: // 4: sum (sum ((U * U) .* U))
            /*
            // using the masked saxpy3 method
            LG_TRY (tricount_prep (NULL, &U, A, msg)) ;
            GrB_TRY (GrB_mxm (C, U, NULL, semiring, U, U, GrB_DESC_S)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
             */
            break ;

        default:
        case LAGraph_TriangleCount_SandiaDot: // 5: sum (sum ((L * U') .* L))
            /*
            // This tends to be the fastest method for most large matrices, but
            // the SandiaDot2 method is also very fast.

            // using the masked dot product
            LG_TRY (tricount_prep (&L, &U, A, msg)) ;
            GrB_TRY (GrB_mxm (C, L, NULL, semiring, L, U, GrB_DESC_ST1)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            */
            break ;

        case LAGraph_TriangleCount_SandiaDot2: // 6: sum (sum ((U * L') .* U))
            /*
            // using the masked dot product
            LG_TRY (tricount_prep (&L, &U, A, msg)) ;
            GrB_TRY (GrB_mxm (C, U, NULL, semiring, U, L, GrB_DESC_ST1)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            */
            break ;
    }

    //LG_FREE_ALL ;
    (*ntriangles) = (uint64_t) ntri ;
    return (GrB_SUCCESS) ;
}

//#undef  LG_FREE_ALL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
