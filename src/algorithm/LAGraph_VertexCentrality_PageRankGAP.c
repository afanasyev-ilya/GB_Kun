//------------------------------------------------------------------------------
// LAGraph_VertexCentrality_PageRankGAP: pagerank for the GAP benchmark
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// Contributed by Tim Davis and Mohsen Aznaveh, Texas A&M University.

//------------------------------------------------------------------------------

// PageRank for the GAP benchmark.  This is an "expert" method.

// This algorithm follows the specification given in the GAP Benchmark Suite:
// https://arxiv.org/abs/1508.03619 which assumes that both A and A' are
// already available, as are the row and column degrees.

// The G->AT and G->rowdegree properties must be defined for this method.  If G
// is undirected or G->A is known to have a symmetric structure, then G->A is
// used instead of G->AT, however.

#define LAGraph_FREE_WORK           \
{                                   \
    GrB_free (&d1) ;                \
    GrB_free (&d) ;                 \
    GrB_free (&t) ;                 \
    GrB_free (&w) ;                 \
}

#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)    \
GrB_Index nvals = 0;                                                                 \
GrB_Matrix_nvals(&nvals, matrix);                                                    \
/*printf("matrix has %ld\n edges", nvals);*/                                         \
double t1 = omp_get_wtime();                                                         \
call_instruction;                                                                    \
double t2 = omp_get_wtime();                                                         \
double time = (t2 - t1)*1000;                                                        \
double perf = nvals * 2.0 / ((t2 - t1)*1e9);                                         \
double bw = nvals * bytes_per_flop/((t2 - t1)*1e9);                                  \
/*printf("edges: %lf\n", nvals);*/                                                   \
printf("%s time %lf (ms)\n", op_name, (t2-t1)*1000);                             \
printf("%s perf %lf (GFLop/s)\n", op_name, perf);                                \
/*printf("%s BW %lf (GB/s)\n", op_name, bw);*/                                       \
FILE *f;                                                                             \
f = fopen("perf_stats.txt", "a");                                                    \
fprintf(f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", op_name, time, perf, bw, nvals);\
fclose(f);                                                                           \


#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_FREE_WORK ;             \
    GrB_free (&r) ;                 \
}

#include "LG_internal.h"

int LAGraph_VertexCentrality_PageRankGAP // returns -1 on failure, 0 on success
(
    // outputs:
    GrB_Vector *centrality, // centrality(i): GAP-style pagerank of node i
    // inputs:
    LAGraph_Graph G,        // input graph
    float damping,          // damping factor (typically 0.85)
    float tol,              // stopping tolerance (typically 1e-4) ;
    int itermax,            // maximum number of iterations (typically 100)
    int *iters,             // output: number of iterations taken
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector r = NULL, d = NULL, t = NULL, w = NULL, d1 = NULL ;
    LG_CHECK (centrality == NULL, -1, "centrality is NULL") ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), -1, "graph is invalid") ;
    LAGraph_Kind kind = G->kind ; 
    int A_sym_structure = G->A_structure_is_symmetric ;
    GrB_Matrix AT ;
    if (kind == LAGRAPH_ADJACENCY_UNDIRECTED || A_sym_structure == LAGRAPH_TRUE)
    {
        // A and A' have the same structure
        AT = G->A ;
    }
    else
    {
        // A and A' differ
        AT = G->AT ;
        LG_CHECK (AT == NULL, -1, "G->AT is required") ;
    }
    GrB_Vector d_out = G->rowdegree ;
    LG_CHECK (d_out == NULL, -1, "G->rowdegree is required") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    (*centrality) = NULL ;
    GrB_TRY (GrB_Matrix_nrows (&n, AT)) ;

    const float teleport = (1 - damping) / n ;
    float rdiff = 1 ;       // first iteration is always done

    // r = 1 / n
    GrB_TRY (GrB_Vector_new (&t, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&r, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&w, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (r, NULL, NULL, 1.0 / n, GrB_ALL, n, NULL)) ;

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GrB_TRY (GrB_apply (d, NULL, NULL, GrB_DIV_FP32, d_out, damping, NULL)) ;

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;
    GrB_free (&d1) ;

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    for ((*iters) = 0 ; (*iters) < itermax /* && rdiff > tol */; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GrB_TRY (GrB_eWiseMult (w, NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, NULL, NULL, teleport, GrB_ALL, n, NULL)) ;
        
        //GrB_Index nvals = 0;
        //GrB_Matrix_nvals(&nvals, AT);
        GrB_Index v_nvals = 0, v_size;
        GrB_Vector_nvals(&v_nvals, w);
        GrB_Vector_size(&v_size, w);
        // r += A'*w
        
        SAVE_STATS((GrB_TRY (GrB_mxv (r, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
            AT, w, NULL))), "pr_mxv", (sizeof(float)*2 + sizeof(size_t)), 1, AT);

        // t -= r
        GrB_TRY (GrB_assign (t, NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GrB_TRY (GrB_apply (t, NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;
        float ranks_sum = 0;
        GrB_TRY (GrB_reduce (&ranks_sum, NULL, GrB_PLUS_MONOID_FP32, r, NULL));
        //cout << "ranks sum : " << ranks_sum << endl;
        printf("ranks sum : %f\n", ranks_sum);
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*centrality) = r ;
    LAGraph_FREE_WORK ;
    return (0) ;
}

