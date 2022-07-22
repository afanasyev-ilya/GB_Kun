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

#define SSSP_DEBUG
#define YELLOW_BOLD "\033[1;33m"
#define CLEAR "\033[0m"
#define DEBUG_PRINT(function) {cout << YELLOW_BOLD << "[DEBUG] line " << __LINE__ << ": "; function; cout << CLEAR << endl;}

namespace lablas {
namespace algorithm {

template <typename T>
int LAGr_SingleSourceShortestPath
(
    // output:
    lablas::Vector<T> *path_length, // path_length (i) is the length of the shortest
                                    // path from the source vertex to vertex i
    // input:
    const LAGraph_Graph<T>* G,      // input graph, not modified
    GrB_Index source,               // source vertex
    T Delta,                        // delta value for delta stepping
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------    

    lablas::Matrix<T> *A = G->A ;
    GrB_Index n = A->nrows();

    LG_CLEAR_MSG ;
    T                    lBound;    // the threshold for GrB_select
    T                    uBound;    // the threshold for GrB_select
    lablas::Matrix<T>    AL;        // graph containing the light weight edges
    lablas::Matrix<T>    AH;        // graph containing the heavy weight edges
    lablas::Vector<T>    t(n);      // tentative shortest path length
    lablas::Vector<T>    tmasked(n);
    lablas::Vector<T>    tReq(n);
    lablas::Vector<bool> tless(n);
    lablas::Vector<bool> s(n);
    lablas::Vector<bool> reach(n);
    lablas::Vector<bool> Empty(n);
    
#ifdef SSSP_DEBUG

    //--------------------------------------------------------------------------
    // Debug information. Delete on release
    //--------------------------------------------------------------------------

    t.set_name("t");
    tmasked.set_name("tmasked");
    tReq.set_name("tReq");
    tless.set_name("tless");
    s.set_name("s");
    reach.set_name("reach");
    Empty.set_name("Empty");

#endif

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // select the operators, and set t (:) = infinity
    auto ne = [](auto x, auto, auto, auto s) { return x != s; },
         le = [](auto x, auto, auto, auto s) { return x <= s; },
         ge = [](auto x, auto, auto, auto s) { return x >= s; },
         lt = [](auto x, auto, auto, auto s) { return x < s;  },
         gt = [](auto x, auto, auto, auto s) { return x > s;  };

    auto less_than = [](auto x, auto y) { return x < y; };

    auto min_plus = lablas::MinimumPlusSemiring<T>();

    GrB_assign (&t, MASK_VECTOR_NULL, lablas::second<T>(), std::numeric_limits<T>::max(), GrB_ALL, n, NULL);
    
#ifdef SSSP_DEBUG
    DEBUG_PRINT(t.print());
#endif

    // t (src) = 0
    t.set_element(0, source);
#ifdef SSSP_DEBUG
    DEBUG_PRINT(t.print());
#endif

    // reach (src) = true
    reach.set_element(true, source);
#ifdef SSSP_DEBUG
    DEBUG_PRINT(reach.print());
#endif

    // s (src) = true
    s.set_element(true, source);
#ifdef SSSP_DEBUG
    DEBUG_PRINT(s.print());
#endif

    // AL = A .* (A <= Delta)
    GrB_select (&AL, MASK_MATRIX_NULL, NULL, le, A, Delta, NULL);
#ifdef SSSP_DEBUG
    DEBUG_PRINT(AL.print());
#endif

    // FUTURE: costly for some problems, taking up to 50% of the total time:
    // AH = A .* (A > Delta)
    GrB_select (&AH, MASK_MATRIX_NULL, NULL, gt, A, Delta, NULL);
#ifdef SSSP_DEBUG
    DEBUG_PRINT(AH.print());
#endif

    //--------------------------------------------------------------------------
    // while (t >= step*Delta) not empty
    //--------------------------------------------------------------------------

    lablas::Descriptor desc;

    for (int64_t step = 0 ; ; step++)
    {
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << " = = = Iteration " << step << " = = = ");
#endif

        //----------------------------------------------------------------------
        // tmasked = all entries in t<reach> that are less than (step+1)*Delta
        //----------------------------------------------------------------------

        uBound = (step + 1) * Delta;
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "uBound = " << uBound);
#endif

        tmasked.clear();
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "tmasked cleared");
#endif

        // tmasked<reach> = t
        // FUTURE: this is costly, typically using Method 06s in SuiteSparse,
        // which is a very general-purpose one.  Write a specialized kernel to
        // exploit the fact that reach and t are bitmap and tmasked starts
        // empty, or fuse this assignment with the GrB_select below.
        GrB_assign (&tmasked, &reach, NULL, &t, GrB_ALL, n, &desc);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(tmasked.print());
#endif
    
        // tmasked = select (tmasked < (step+1)*Delta)
        GrB_select (&tmasked, MASK_VECTOR_NULL, NULL, lt, &tmasked, uBound, NULL);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(tmasked.print());
#endif
        // --- alternative:
        // FUTURE this is slower than the above but should be much faster.
        // GrB_select is computing a bitmap result then converting it to
        // sparse.  t and reach are both bitmap and tmasked finally sparse.
        // tmasked<reach> = select (t < (step+1)*Delta)
        // GrB_select (tmasked, reach, NULL, lt, t, uBound, NULL)) ;

        GrB_Index tmasked_nvals = tmasked.nvals();

        //----------------------------------------------------------------------
        // continue while the current bucket (tmasked) is not empty
        //----------------------------------------------------------------------

        while (tmasked_nvals > 0)
        {
            // tReq = AL'*tmasked using the min_plus semiring
            GrB_vxm (&tReq, MASK_VECTOR_NULL, NULL, min_plus, &tmasked, &AL, NULL);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(tReq.print());
#endif

            // s<struct(tmasked)> = true
            GrB_assign (&s, &tmasked, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(s.print());
#endif

            // if nvals (tReq) is 0, no need to continue the rest of this loop
            GrB_Index tReq_nvals = tReq.nvals();
#ifdef SSSP_DEBUG
            DEBUG_PRINT(cout << "tReq_nvals = " << tReq_nvals);
#endif
            if (tReq_nvals == 0) break ;

            // tless = (tReq .< t) using set intersection
            GrB_eWiseMult (&tless, MASK_VECTOR_NULL, NULL, less_than, &tReq, &t, NULL);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(tless.print());
#endif

            // remove explicit zeros from tless so it can be used as a
            // structural mask
            GrB_select (&tless, MASK_VECTOR_NULL, NULL, ne, &tless, 0, NULL);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(tless.print());
#endif
        
            GrB_Index tless_nvals = tless.nvals();
#ifdef SSSP_DEBUG
            DEBUG_PRINT(cout << "tless_nvals = " << tless_nvals);
#endif
            if (tless_nvals == 0) break ;

            // update reachable node list/mask
            // reach<struct(tless)> = true
            GrB_assign (&reach, &tless, NULL, (bool) true, GrB_ALL, n, GrB_DESC_S);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(reach.print());
#endif         

            // tmasked<struct(tless)> = select (tReq < (step+1)*Delta)
            tmasked.clear();
#ifdef SSSP_DEBUG
            DEBUG_PRINT(cout << "tmasked cleared");
#endif

            GrB_select (&tmasked, &tless, NULL, lt, &tReq, uBound, GrB_DESC_S);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(tmasked.print());
#endif
            // t<struct(tless)> = tReq
            GrB_assign (&t, &tless, NULL, &tReq, GrB_ALL, n, GrB_DESC_S);
#ifdef SSSP_DEBUG
            DEBUG_PRINT(t.print());
#endif
            tmasked_nvals = tmasked.nvals();
#ifdef SSSP_DEBUG
            DEBUG_PRINT(cout << "tmasked_nvals = " << tmasked_nvals);
#endif
        }

        // tmasked<s> = t
        tmasked.clear();
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "tmasked cleared");
#endif

        GrB_assign (&tmasked, &s, NULL, &t, GrB_ALL, n, GrB_DESC_S);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(tmasked.print());
#endif

        // tReq = AH'*tmasked using the min_plus semiring
        GrB_vxm (&tReq, MASK_VECTOR_NULL, NULL, min_plus, &tmasked, &AH, NULL);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(tReq.print());
#endif

        // tless = (tReq .< t) using set intersection
        GrB_eWiseMult (&tless, MASK_VECTOR_NULL, NULL, less_than, &tReq, &t, NULL);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(tless.print());
#endif

        // t<tless> = tReq, which computes t = min (t, tReq)
        GrB_assign (&t, &tless, NULL, &tReq, GrB_ALL, n, &desc);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(t.print());
#endif

        //----------------------------------------------------------------------
        // find out how many left to be computed
        //----------------------------------------------------------------------

        // update reachable node list
        // reach<tless> = true
        GrB_assign (&reach, &tless, NULL, (bool) true, GrB_ALL, n, &desc);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(reach.print());
#endif

        // remove previous buckets
        // reach<struct(s)> = Empty
        GrB_assign (&reach, &s, NULL, &Empty, GrB_ALL, n, GrB_DESC_S);
#ifdef SSSP_DEBUG
        DEBUG_PRINT(reach.print());
#endif

        GrB_Index nreach = reach.nvals();
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "reach_nvals = " << nreach);
#endif
        if (nreach == 0) break ;

        s.clear(); // clear s for the next iteration
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "s cleared");
#endif
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    t.dup(path_length) ;
#ifdef SSSP_DEBUG
        DEBUG_PRINT(cout << "Final path lengths: "; t.print());
#endif
    return (GrB_SUCCESS) ;
}

}
}
