/// @file tc.hpp
/// @author Lastname:Firstname:A00123456:cscxxxxx
/// @version Revision 1.1
/// @brief TC algorithm.
/// @details code taken form LAGraph. Add license.
/// @date May 12, 2022

#pragma once
#include "../../src/gb_kun.h"

#define GrB_Matrix lablas::Matrix<int>*
#define GrB_Vector lablas::Vector<int>*

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @namespace Lablas

namespace lablas {

/// @namespace Algorithm

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

#define LG_FREE_ALL             \
{                               \
    GrB_free (L) ;              \
    GrB_free (U) ;              \
}

/// @brief Triangle count preparation function for TC algorithms.
///
/// This algorithm implements LU-decomposition for several TC algorithms by using GrB_select function.
/// @param[out] L Pointer to the first (left) selected matrix from LU-decomposition
/// @param[out] U Pointer to the second (right) selected matrix from LU-decomposition
/// @param[in] A Input matrix
/// @result GrB status
static int tricount_prep(GrB_Matrix *L,      // if present, compute L = tril (A,-1)
                         GrB_Matrix *U,      // if present, compute U = triu (A, 1)
                         GrB_Matrix A)
{
    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, A)) ;

    if (L != NULL)
    {
        // L = tril (A,-1)
        GrB_TRY (GrB_Matrix_new (L, GrB_BOOL, n, n)) ;
        auto GrB_TRIL_select = [](int x, Index i, Index j, int val){
            return (j <= (i + val)) ? 0 : x;
        };
        #define MASK_NULL static_cast<const lablas::Matrix<float>*>(NULL)
        GrB_TRY (GrB_select (*L, MASK_NULL, NULL, GrB_TRIL_select, A, (int64_t) (-1), NULL)) ;
        #undef MASK_NULL
    }

    if (U != NULL)
    {
        // U = triu (A,1)
        GrB_TRY (GrB_Matrix_new (U, GrB_BOOL, n, n)) ;
        auto GrB_TRIU_select = [](int x, Index i, Index j, int val){
            return (j >= (i + val)) ? 0 : x;
        };
        #define MASK_NULL static_cast<const lablas::Matrix<float>*>(NULL)
        GrB_TRY (GrB_select (*U, MASK_NULL, NULL, GrB_TRIU_select, A, (int64_t) 1, NULL)) ;
        #undef MASK_NULL
    }
    return (GrB_SUCCESS) ;
}
#undef  LG_FREE_ALL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LG_FREE_ALL                         \
{                                           \
    GrB_free (&C) ;                         \
    GrB_free (&L) ;                         \
    GrB_free (&T) ;                         \
    GrB_free (&U) ;                         \
}

/// @brief Triangle count function.
///
/// This algorithm implements triangle counting by using several TC algorithms implementations.
/// Implemented TC algorithms are as follows:
/// <li> Burkhardt algorithm (used by default),
/// <li> Cohen algorithm,
/// <li> Sandia algorithm,
/// <li> SandiaDot algorithm
/// By using descriptor it is also possible to choose from mxm methods to be used in TC.
/// @param[out] ntriangles Pointer to the triangles count result variable
/// @param[in] G Pointer to the input Graph object
/// @param[in] method Lagraph method enum value
/// @param[in] presort Lagraph presort option enum value
/// @param[in] desc Pointer to the descriptor
/// @result GrB status
int LAGr_TriangleCount(uint64_t *ntriangles, const LAGraph_Graph<int>* G,
                       LAGraph_TriangleCount_Method method,
                       LAGraph_TriangleCount_Presort presort,
                       Descriptor *desc)
{
    LG_CLEAR_MSG ;
    GrB_Matrix C = NULL;
    GrB_Matrix L = NULL;
    GrB_Matrix U = NULL;
    GrB_Matrix T = NULL;
    int64_t *P = NULL ;

    assert(method == LAGraph_TriangleCount_Default ||   // 0: use default method
           method == LAGraph_TriangleCount_Burkhardt || // 1: sum (sum ((A^2) .* A))/6
           method == LAGraph_TriangleCount_Cohen ||     // 2: sum (sum ((L * U) .*A))/2
           method == LAGraph_TriangleCount_Sandia ||    // 3: sum (sum ((L * L) .* L))
           method == LAGraph_TriangleCount_Sandia2 ||   // 4: sum (sum ((U * U) .* U))
           method == LAGraph_TriangleCount_SandiaDot || // 5: sum (sum ((L * U') .* L))
           method == LAGraph_TriangleCount_SandiaDot2  // 6: sum (sum ((U * L') .* U))
           );
    assert((presort) == LAGraph_TriangleCount_NoSort ||
           (presort) == LAGraph_TriangleCount_Ascending ||
           (presort) == LAGraph_TriangleCount_Descending ||
           (presort) == LAGraph_TriangleCount_AutoSort);

    if (method == LAGraph_TriangleCount_Default) {
        method = LAGraph_TriangleCount_Burkhardt ;
    }

    // the Sandia* methods can benefit from the presort
    bool method_can_use_presort =
            method == LAGraph_TriangleCount_Sandia ||
            method == LAGraph_TriangleCount_Sandia2 ||
            method == LAGraph_TriangleCount_SandiaDot ||
            method == LAGraph_TriangleCount_SandiaDot2;

    GrB_Matrix A = G->A ;
    auto Degree = G->rowdegree;

    bool auto_sort = (presort == LAGraph_TriangleCount_AutoSort);
    if (auto_sort && method_can_use_presort) {
        assert(Degree != NULL && "G->rowdegree is required") ;
    }

    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GrB_TRY (GrB_Matrix_new (&C, GrB_INT64, n, n)) ;
    auto semiring = LAGraph_plus_one_int64 ;
    auto monoid = GrB_PLUS_MONOID_INT64 ;

    int64_t ntri ;

    Desc_value multiplication_mode = GrB_IKJ_MASKED;
    if (desc) {
        backend::Descriptor* desc_t =  desc->get_descriptor();
        desc_t->get(GrB_MXMMODE, &multiplication_mode);
    }

    Descriptor mxm_desc;
    if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
        mxm_desc = GrB_DESC_IJK_DOUBLE_SORT;
    } else {
        mxm_desc = GrB_DESC_IKJ_MASKED;
    }

    // For now the most fast and stable TC algorithm is LAGraph_TriangleCount_Burkhardt
    switch (method)
    {
        default:
        case LAGraph_TriangleCount_Burkhardt:  // 1: sum (sum ((A^2) .* A)) / 6
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                A->get_matrix()->sort_csr_columns("STL_SORT");
                A->get_matrix()->sort_csc_rows("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, A, NULL, semiring, A, A, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_Cohen: // 2: sum (sum ((L * U) .* A)) / 2
            GrB_TRY (static_cast<LA_Info>(tricount_prep(&L, &U, A))) ;
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                L->get_matrix()->sort_csr_columns("STL_SORT");
                U->get_matrix()->sort_csc_rows("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, A, NULL, semiring, L, U, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_Sandia: // 3: sum (sum ((L * L) .* L))
            // using the masked saxpy3 method
            GrB_TRY (static_cast<LA_Info>(tricount_prep(&L, NULL, A))) ;
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                L->get_matrix()->sort_csr_columns("STL_SORT");
                L->get_matrix()->sort_csc_rows("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, L, NULL, semiring, L, L, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_Sandia2: // 4: sum (sum ((U * U) .* U))
            // using the masked saxpy3 method
            GrB_TRY (static_cast<LA_Info>(tricount_prep(NULL, &U, A))) ;
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                U->get_matrix()->sort_csr_columns("STL_SORT");
                U->get_matrix()->sort_csc_rows("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, U, NULL, semiring, U, U, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_SandiaDot: // 5: sum (sum ((L * U') .* L))
            // This tends to be the fastest method for most large matrices, but
            // the SandiaDot2 method is also very fast.
            // using the masked dot product
            GrB_TRY (static_cast<LA_Info>(tricount_prep(&L, &U, A))) ;
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                U->get_matrix()->sort_csc_rows("STL_SORT");
                L->get_matrix()->sort_csr_columns("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, L, NULL, semiring, L, U, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;

        case LAGraph_TriangleCount_SandiaDot2: // 6: sum (sum ((U * L') .* U))
            // using the masked dot product
            GrB_TRY (static_cast<LA_Info>(tricount_prep(&L, &U, A))) ;
            if (multiplication_mode == GrB_IJK_DOUBLE_SORT) {
                U->get_matrix()->sort_csr_columns("STL_SORT");
                L->get_matrix()->sort_csc_rows("STL_SORT");
            }
            GrB_TRY (GrB_mxm (C, U, NULL, semiring, U, L, &mxm_desc)) ;
            GrB_TRY (GrB_reduce (&ntri, NULL, monoid, C, NULL)) ;
            ntri /= 6 ;
            break ;
    }

    LG_FREE_ALL ;
    (*ntriangles) = (uint64_t) ntri ;
    return (GrB_SUCCESS) ;
}

#undef  LG_FREE_ALL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
