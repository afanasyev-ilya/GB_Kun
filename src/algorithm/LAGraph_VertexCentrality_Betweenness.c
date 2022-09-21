//------------------------------------------------------------------------------
// LAGraph_VertexCentrality_Betweenness: vertex betweenness-centrality
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// Contributed by Scott Kolodziej and Tim Davis, Texas A&M University;
// Adapted and revised from GraphBLAS C API Spec, Appendix B.4.

//------------------------------------------------------------------------------

// LAGraph_VertexCentrality_Betweenness: Batch algorithm for computing
// betweeness centrality, using push-pull optimization.

// This method computes an approximation of the betweenness algorithm.
//                               ____
//                               \      sigma(s,t | i)
//    Betweenness centrality =    \    ----------------
//           of node i            /       sigma(s,t)
//                               /___
//                            s != i != t
//
// Where sigma(s,t) is the total number of shortest paths from node s to
// node t, and sigma(s,t | i) is the total number of shortest paths from
// node s to node t that pass through node i.
//
// Note that the true betweenness centrality requires computing shortest paths
// from all nodes s to all nodes t (or all-pairs shortest paths), which can be
// expensive to compute. By using a reasonably sized subset of source nodes, an
// approximation can be made.
//
// This method performs simultaneous breadth-first searches of the entire graph
// starting at a given set of source nodes. This pass discovers all shortest
// paths from the source nodes to all other nodes in the graph.  After the BFS
// is complete, the number of shortest paths that pass through a given node is
// tallied by reversing the traversal. From this, the (approximate) betweenness
// centrality is computed.

// G->A represents the graph, and G->AT must be present.  G->A must be square,
// and can be unsymmetric.  Self-edges are OK.  The values of G->A and G->AT
// are ignored; just the structure of two matrices are used.

// Each phase uses push-pull direction optimization.

// This is an LAGraph "expert" method, since it requires the source nodes to be
// specified, and G->AT must be present (unless G is undirected or G->A is
// known to have a symmetric structure, in which case G->A is used for both A
// and AT).

//------------------------------------------------------------------------------

#define LAGraph_FREE_WORK                       \
{                                               \
    GrB_free (&frontier) ;                      \
    GrB_free (&paths) ;                         \
    GrB_free (&bc_update) ;                     \
    GrB_free (&W) ;                             \
    if (S != NULL)                              \
    {                                           \
        for (int64_t i = 0 ; i < n ; i++)       \
        {                                       \
            if (S [i] == NULL) break ;          \
            GrB_free (&(S [i])) ;               \
        }                                       \
        LAGraph_Free ((void **) &S) ;           \
    }                                           \
}

#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_FREE_WORK ;             \
    GrB_free (centrality) ;         \
}

#define SAVE_TIME(call_instruction, op_name)       \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                           \
double my_perf = 0;                                        \
double my_bw = 0;                                  \
size_t my_nvals = 0;\
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");                                                 \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);\


#include "LG_internal.h"

//------------------------------------------------------------------------------
// LAGraph_VertexCentrality_Betweenness: vertex betweenness-centrality
//------------------------------------------------------------------------------

int LAGraph_VertexCentrality_Betweenness    // vertex betweenness-centrality
(
    // outputs:
    GrB_Vector *centrality,     // centrality(i): betweeness centrality of i
    // inputs:
    LAGraph_Graph G,            // input graph
    const GrB_Index *sources,   // source vertices to compute shortest paths
    int32_t ns,                 // number of source vertices
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;

    // Array of BFS search matrices.
    // S [i] is a sparse matrix that stores the depth at which each vertex is
    // first seen thus far in each BFS at the current depth i. Each column
    // corresponds to a BFS traversal starting from a source node.
    GrB_Matrix *S = NULL ;

    // Frontier matrix, a sparse matrix.
    // Stores # of shortest paths to vertices at current BFS depth
    GrB_Matrix frontier = NULL ;

    // Paths matrix holds the number of shortest paths for each node and
    // starting node discovered so far.  A dense matrix that is updated with
    // sparse updates, and also used as a mask.
    GrB_Matrix paths = NULL ;

    // Update matrix for betweenness centrality, values for each node for
    // each starting node.  A dense matrix.
    GrB_Matrix bc_update = NULL ;

    // Temporary workspace matrix (sparse).
    GrB_Matrix W = NULL ;

    GrB_Index n = 0 ;                   // # nodes in the graph

    LG_CHECK (centrality == NULL, -1, "centrality is NULL") ;
    (*centrality) = NULL ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), -1, "graph is invalid") ;
    LAGraph_Kind kind = G->kind ;
    int A_sym_structure = G->A_structure_is_symmetric ;

    GrB_Matrix A = G->A ;
    GrB_Matrix AT ;
    if (kind == LAGRAPH_ADJACENCY_UNDIRECTED || A_sym_structure == LAGRAPH_TRUE)
    {
        // A and A' have the same structure
        AT = A ;
    }
    else
    {
        // A and A' differ
        AT = G->AT ;
        LG_CHECK (AT == NULL, -1, "G->AT is required") ;
    }

    // =========================================================================
    // === initializations =====================================================
    // =========================================================================

    // Initialize paths and frontier with source notes
    GrB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GrB_TRY (GrB_Matrix_new (&paths,    GrB_FP64, ns, n)) ;
    GrB_TRY (GrB_Matrix_new (&frontier, GrB_FP64, ns, n)) ;
    #if LG_SUITESPARSE
    GrB_TRY (GxB_set (paths, GxB_SPARSITY_CONTROL, GxB_BITMAP + GxB_FULL)) ;
    #endif
    for (GrB_Index i = 0 ; i < ns ; i++)
    {
        // paths (i,s(i)) = 1
        // frontier (i,s(i)) = 1
        double one = 1 ;
        GrB_TRY (GrB_Matrix_setElement (paths,    one, i, sources [i])) ;
        GrB_TRY (GrB_Matrix_setElement (frontier, one, i, sources [i])) ;
    }

    // Initial frontier: frontier<!paths>= frontier*A
    SAVE_TIME((
    GrB_TRY (GrB_mxm (frontier, paths, NULL, LAGraph_plus_first_fp64,
        frontier, A, GrB_DESC_RSC))), "BC_mxm");

    // Allocate memory for the array of S matrices
    S = (GrB_Matrix *) LAGraph_Malloc (n+1, sizeof (GrB_Matrix)) ;
    LG_CHECK (S == NULL, -1, "out of memory") ;
    S [0] = NULL ;

    // =========================================================================
    // === Breadth-first search stage ==========================================
    // =========================================================================

    bool last_was_pull = false ;
    GrB_Index frontier_size, last_frontier_size = 0 ;
    GrB_TRY (GrB_Matrix_nvals (&frontier_size, frontier)) ;

    int64_t depth ;
    for (depth = 0 ; frontier_size > 0 && depth < n ; depth++)
    {

        //----------------------------------------------------------------------
        // S [depth] = structure of frontier
        //----------------------------------------------------------------------

        S [depth+1] = NULL ;
        LAGraph_TRY (LAGraph_Structure (&(S [depth]), frontier, msg)) ;

        //----------------------------------------------------------------------
        // Accumulate path counts: paths += frontier
        //----------------------------------------------------------------------

        GrB_TRY (GrB_assign (paths, NULL, GrB_PLUS_FP64, frontier, GrB_ALL, ns,
            GrB_ALL, n, NULL)) ;

        //----------------------------------------------------------------------
        // Update frontier: frontier<!paths> = frontier*A
        //----------------------------------------------------------------------

        // pull if frontier is more than 10% dense,
        // or > 6% dense and last step was pull
        double frontier_density = ((double) frontier_size) / (double) (ns*n) ;
        bool do_pull = frontier_density > (last_was_pull ? 0.06 : 0.10 ) ;

        if (do_pull)
        {
            // frontier<!paths> = frontier*AT'
            #if LG_SUITESPARSE
            GrB_TRY (GxB_set (frontier, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            #endif
            SAVE_TIME((GrB_TRY (GrB_mxm (frontier, paths, NULL, LAGraph_plus_first_fp64,
                frontier, AT, GrB_DESC_RSCT1))), "mxm_pull");
        }
        else // push
        {
            // frontier<!paths> = frontier*A
            #if LG_SUITESPARSE
            GrB_TRY (GxB_set (frontier, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            #endif
            SAVE_TIME((GrB_TRY (GrB_mxm (frontier, paths, NULL, LAGraph_plus_first_fp64,
                frontier, A, GrB_DESC_RSC))), "mxm_push");
        }

        //----------------------------------------------------------------------
        // Get size of current frontier: frontier_size = nvals(frontier)
        //----------------------------------------------------------------------

        last_frontier_size = frontier_size ;
        last_was_pull = do_pull ;
        GrB_TRY (GrB_Matrix_nvals (&frontier_size, frontier)) ;
    }

    GrB_TRY (GrB_free (&frontier)) ;

    // =========================================================================
    // === Betweenness centrality computation phase ============================
    // =========================================================================

    // bc_update = ones (ns, n) ; a full matrix (and stays full)
    GrB_TRY (GrB_Matrix_new (&bc_update, GrB_FP64, ns, n)) ;
    GrB_TRY (GrB_assign (bc_update, NULL, NULL, 1, GrB_ALL, ns, GrB_ALL, n,
        NULL)) ;
    // W: empty ns-by-n array, as workspace
    GrB_TRY (GrB_Matrix_new (&W, GrB_FP64, ns, n)) ;

    // Backtrack through the BFS and compute centrality updates for each vertex
    for (int64_t i = depth-1 ; i > 0 ; i--)
    {

        //----------------------------------------------------------------------
        // W<S[i]> = bc_update ./ paths
        //----------------------------------------------------------------------

        // Add contributions by successors and mask with that level's frontier
        GrB_TRY (GrB_eWiseMult (W, S [i], NULL, GrB_DIV_FP64, bc_update, paths,
            GrB_DESC_RS)) ;

        //----------------------------------------------------------------------
        // W<S[i−1]> = W * A'
        //----------------------------------------------------------------------

        // pull if W is more than 10% dense and nnz(W)/nnz(S[i-1]) > 1
        // or if W is more than 1% dense and nnz(W)/nnz(S[i-1]) > 10
        GrB_Index wsize, ssize ;
        GrB_Matrix_nvals (&wsize, W) ;
        GrB_Matrix_nvals (&ssize, S [i-1]) ;
        double w_density    = ((double) wsize) / ((double) (ns*n)) ;
        double w_to_s_ratio = ((double) wsize) / ((double) ssize) ;
        bool do_pull = (w_density > 0.1  && w_to_s_ratio > 1.) ||
                       (w_density > 0.01 && w_to_s_ratio > 10.) ;

        if (do_pull)
        {
            // W<S[i−1]> = W * A'
            #if LG_SUITESPARSE
            GrB_TRY (GxB_set (W, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            #endif
            GrB_TRY (GrB_mxm (W, S [i-1], NULL, LAGraph_plus_first_fp64, W, A,
                GrB_DESC_RST1)) ;
        }
        else // push
        {
            // W<S[i−1]> = W * AT
            #if LG_SUITESPARSE
            GrB_TRY (GxB_set (W, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            #endif
            GrB_TRY (GrB_mxm (W, S [i-1], NULL, LAGraph_plus_first_fp64, W, AT,
                GrB_DESC_RS)) ;
        }

        //----------------------------------------------------------------------
        // bc_update += W .* paths
        //----------------------------------------------------------------------

        GrB_TRY (GrB_eWiseMult (bc_update, NULL, GrB_PLUS_FP64, GrB_TIMES_FP64,
            W, paths, NULL)) ;
    }

    // =========================================================================
    // === finalize the centrality =============================================
    // =========================================================================

    // Initialize the centrality array with -ns to avoid counting
    // zero length paths
    GrB_TRY (GrB_Vector_new (centrality, GrB_FP64, n)) ;
    GrB_TRY (GrB_assign (*centrality, NULL, NULL, -ns, GrB_ALL, n, NULL)) ;

    // centrality (i) = sum (bc_update (:,i)) for all nodes i
    GrB_TRY (GrB_reduce (*centrality, NULL, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64,
        bc_update, GrB_DESC_T0)) ;

    LAGraph_FREE_WORK ;
    return (0) ;
}
