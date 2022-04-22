//------------------------------------------------------------------------------
// LAGr_SingleSourceShortestPath: single-source shortest path
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

// Contributed by Jinhao Chen, Scott Kolodziej and Tim Davis, Texas A&M
// University.  Adapted from GraphBLAS Template Library (GBTL) by Scott
// McMillan and Tze Meng Low.

//------------------------------------------------------------------------------

// This is an Advanced algorithm (G->emin is required).

// Single source shortest path with delta stepping.

// U. Sridhar, M. Blanco, R. Mayuranath, D. G. Spampinato, T. M. Low, and
// S. McMillan, "Delta-Stepping SSSP: From Vertices and Edges to GraphBLAS
// Implementations," in 2019 IEEE International Parallel and Distributed
// Processing Symposium Workshops (IPDPSW), 2019, pp. 241–250.
// https://ieeexplore.ieee.org/document/8778222/references
// https://arxiv.org/abs/1911.06895

// LAGr_SingleSourceShortestPath computes the shortest path lengths from the
// specified source vertex to all other vertices in the graph.

// The parent vector is not computed; see LAGraph_BF_* instead.

// NOTE: this method gets stuck in an infinite loop when there are negative-
// weight cycles in the graph.

// FUTURE: a Basic algorithm that picks Delta automatically

#pragma once

#define MASK_VECTOR_NULL static_cast<const lablas::Vector<bool>*>(NULL)
#define MASK_MATRIX_NULL static_cast<const lablas::Matrix<bool>*>(NULL)

namespace lablas {
namespace algorithm {

#define LG_FREE_WORK        \
{                           \
    GrB_free (&AL) ;        \
    GrB_free (&AH) ;        \
    GrB_free (&tmasked) ;   \
    GrB_free (&tReq) ;      \
    GrB_free (&tless) ;     \
    GrB_free (&s) ;         \
    GrB_free (&reach) ;     \
    GrB_free (&Empty) ;     \
}

#define LG_FREE_ALL         \
{                           \
    LG_FREE_WORK ;          \
    GrB_free (&t) ;         \
}

template <typename T>
int LAGr_SingleSourceShortestPath
(
    // output:
    lablas::Vector<T> **path_length,    // path_length (i) is the length of the shortest
                                // path from the source vertex to vertex i
    // input:
    const LAGraph_Graph<T>* G,   // input graph, not modified
    GrB_Index source,            // source vertex
    T Delta,           // delta value for delta stepping
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    T                    *lBound  = NULL ;  // the threshold for GrB_select
    T                    *uBound  = NULL ;  // the threshold for GrB_select
    lablas::Matrix<T>    *AL      = NULL ;  // graph containing the light weight edges
    lablas::Matrix<T>    *AH      = NULL ;  // graph containing the heavy weight edges
    lablas::Vector<T>    *t       = NULL ;  // tentative shortest path length
    lablas::Vector<T>    *tmasked = NULL ;
    lablas::Vector<T>    *tReq    = NULL ;
    lablas::Vector<bool> *tless   = NULL ;
    lablas::Vector<bool> *s       = NULL ;
    lablas::Vector<bool> *reach   = NULL ;
    lablas::Vector<bool> *Empty   = NULL ;

    (*path_length) = NULL ;

    lablas::Matrix<T> *A = G->A ;
    GrB_Index n = A->nrows();

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    lBound  = new T;
    uBound  = new T;
    t       = new Vector<T>(n);
    tmasked = new Vector<T>(n);
    tReq    = new Vector<T>(n);
    Empty   = new Vector<bool>(n);
    tless   = new Vector<bool>(n);
    s       = new Vector<bool>(n);
    reach   = new Vector<bool>(n);


    // select the operators, and set t (:) = infinity
    auto ne = [](auto x, auto, auto, auto s) { return x != s; },
         le = [](auto x, auto, auto, auto s) { return x <= s; },
         ge = [](auto x, auto, auto, auto s) { return x >= s; },
         lt = [](auto x, auto, auto, auto s) { return x < s;  },
         gt = [](auto x, auto, auto, auto s) { return x > s;  };

    auto less_than = [](auto x, auto y) { return x < y; };

    auto min_plus = lablas::MinimumPlusSemiring<T>();

    GrB_assign (t, MASK_VECTOR_NULL, lablas::second<T>(), std::numeric_limits<T>::max(), GrB_ALL, n, NULL);

    // t (src) = 0
    t->set_element(0, source);

    // reach (src) = true
    reach->set_element(true, source);

    // s (src) = true
    s->set_element(true, source);

    // AL = A .* (A <= Delta)
    AL = new Matrix<T>(n, n);
    GrB_select (AL, MASK_MATRIX_NULL, NULL, le, A, Delta, NULL);

    // FUTURE: costly for some problems, taking up to 50% of the total time:
    // AH = A .* (A > Delta)

    AH = new Matrix<T>(n, n);
    GrB_select (AH, MASK_MATRIX_NULL, NULL, gt, A, Delta, NULL);

    //--------------------------------------------------------------------------
    // while (t >= step*Delta) not empty
    //--------------------------------------------------------------------------

    for (int64_t step = 0 ; ; step++)
    {

        //----------------------------------------------------------------------
        // tmasked = all entries in t<reach> that are less than (step+1)*Delta
        //----------------------------------------------------------------------

        *uBound = (step + 1) * Delta;
        tmasked->clear();

        // tmasked<reach> = t
        // FUTURE: this is costly, typically using Method 06s in SuiteSparse,
        // which is a very general-purpose one.  Write a specialized kernel to
        // exploit the fact that reach and t are bitmap and tmasked starts
        // empty, or fuse this assignment with the GrB_select below.
        GrB_assign (tmasked, reach, NULL, t, GrB_ALL, n, NULL);
        // tmasked = select (tmasked < (step+1)*Delta)
        GrB_select (tmasked, MASK_VECTOR_NULL, NULL, lt, tmasked, *uBound, NULL);
        // --- alternative:
        // FUTURE this is slower than the above but should be much faster.
        // GrB_select is computing a bitmap result then converting it to
        // sparse.  t and reach are both bitmap and tmasked finally sparse.
        // tmasked<reach> = select (t < (step+1)*Delta)
        // GrB_select (tmasked, reach, NULL, lt, t, uBound, NULL)) ;

        GrB_Index tmasked_nvals = tmasked->nvals();

        //----------------------------------------------------------------------
        // continue while the current bucket (tmasked) is not empty
        //----------------------------------------------------------------------

        while (tmasked_nvals > 0)
        {
            // tReq = AL'*tmasked using the min_plus semiring
            GrB_vxm (tReq, MASK_VECTOR_NULL, NULL, min_plus, tmasked, AL, NULL);

            // s<struct(tmasked)> = true
            GrB_assign (s, tmasked, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S);

            // if nvals (tReq) is 0, no need to continue the rest of this loop
            GrB_Index tReq_nvals = tReq->nvals();
            if (tReq_nvals == 0) break ;

            // tless = (tReq .< t) using set intersection
            GrB_eWiseMult (tless, MASK_VECTOR_NULL, NULL, less_than, tReq, t, NULL);

            // remove explicit zeros from tless so it can be used as a
            // structural mask
            GrB_Index tless_nvals = tless->nvals();
            GrB_select (tless, MASK_VECTOR_NULL, NULL, ne, tless, 0, NULL);
            if (tless_nvals == 0) break ;

            // update reachable node list/mask
            // reach<struct(tless)> = true
            GrB_assign (reach, tless, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S);

            // tmasked<struct(tless)> = select (tReq < (step+1)*Delta)
            tmasked->clear();
            GrB_select (tmasked, tless, NULL, lt, tReq, *uBound, GrB_DESC_S);

            // t<struct(tless)> = tReq
            GrB_assign (t, tless, NULL, tReq, GrB_ALL, n, GrB_DESC_S);
            tmasked_nvals = tmasked->nvals();
        }

        // tmasked<s> = t
        GrB_Vector_clear (tmasked);
        GrB_assign (tmasked, s, NULL, t, GrB_ALL, n, GrB_DESC_S);

        // tReq = AH'*tmasked using the min_plus semiring
        GrB_vxm (tReq, MASK_VECTOR_NULL, NULL, min_plus, tmasked, AH, NULL);

        // tless = (tReq .< t) using set intersection
        GrB_eWiseMult (tless, MASK_VECTOR_NULL, NULL, less_than, tReq, t, NULL);

        // t<tless> = tReq, which computes t = min (t, tReq)
        GrB_assign (t, tless, NULL, tReq, GrB_ALL, n, NULL);

        //----------------------------------------------------------------------
        // find out how many left to be computed
        //----------------------------------------------------------------------

        // update reachable node list
        // reach<tless> = true
        GrB_assign (reach, tless, NULL, (bool) true, GrB_ALL, n, NULL);

        // remove previous buckets
        // reach<struct(s)> = Empty
        GrB_assign (reach, s, NULL, Empty, GrB_ALL, n, GrB_DESC_S);
        GrB_Index nreach = reach->nvals();
        if (nreach == 0) break ;

        s->clear(); // clear s for the next iteration
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*path_length) = t ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}

}
}
