//------------------------------------------------------------------------------
// test_gappagerank: read in (or create) a matrix and test the GAP PageRank
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause

//------------------------------------------------------------------------------

// Contributed by Tim Davis, Texas A&M and Gabor Szarnyas, BME

#include "LAGraph_demo.h"

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LAGraph_FREE_OUTER                        \
{                                               \
    GrB_free (&A) ;                             \
    GrB_free (&Abool) ;                         \
    GrB_free (&PR) ;                            \
    LAGraph_Delete (&G, msg) ;                  \
    if (f != NULL) fclose (f) ;                 \
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
/*printf("%s perf %lf (GFLop/s)\n", op_name, perf);*/                                \
/*printf("%s BW %lf (GB/s)\n", op_name, bw);*/                                       \
FILE *f;                                                                             \
f = fopen("perf_stats.txt", "a");                                                    \
fprintf(f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", op_name, time, perf, bw, nvals);\
fclose(f);                                                                           \

#define LG_FREE_ALL        \
{                          \
    GrB_free(&d) ;         \
    GrB_free(&dtmp) ;      \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include <LG_internal.h>  // from src/utility

GrB_Info LAGraph_BF_basic_vxm
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix AT,        //transposed adjacency matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols;
    // tmp vector to store distance vector after n loops
    GrB_Vector d = NULL, dtmp = NULL;

    //LG_ASSERT (AT != NULL && pd_output != NULL, GrB_NULL_POINTER) ;

    *pd_output = NULL;
    GrB_TRY (GrB_Matrix_nrows (&nrows, AT)) ;
    GrB_TRY (GrB_Matrix_ncols (&ncols, AT)) ;
    //LG_ASSERT_MSG (nrows == ncols, -1002, "AT must be square") ;
    GrB_Index n = nrows;           // n = # of vertices in graph
    //LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // Initialize distance vector, change the d[s] to 0
    GrB_TRY (GrB_Vector_new(&d, GrB_FP64, n));
    GrB_TRY (GrB_Vector_setElement_FP64(d, 0, s));

    // copy d to dtmp in order to create a same size of vector
    GrB_TRY (GrB_Vector_dup(&dtmp, d));

    int64_t iter = 0;      //number of iterations
    bool same = false;     //variable indicating if d == dtmp

    // terminate when no new path is found or more than n-1 loops
    while (!same && iter < 20)
    {
        // excute semiring on d and AT, and save the result to d
        //SAVE_STATS((GrB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
        //   AT, d, GrB_NULL))), "mxv_sssp", (sizeof(double)*2 + sizeof(size_t)), 1, AT);
        GrB_Index loc_nvals = 0;
        GrB_Vector_nvals (&loc_nvals, d);
        printf("nvals = %lld\n", loc_nvals);
        SAVE_STATS(( GrB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, AT,
            GrB_NULL))), "vxm_sssp", (sizeof(double)*2 + sizeof(size_t)), 1, AT);

        //SAVE_STATS((GrB_TRY (GrB_mxv(dtmp, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
        //    AT, d, GrB_NULL))), "mxv_sssp", (sizeof(float)*2 + sizeof(size_t)), 1, AT);
        LAGraph_Vector_IsEqual (&same, dtmp, d, NULL);
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        printf(same ? "true\n" : "false\n");
        iter++;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (false)
    {
        // excute semiring again to check for negative-weight cycle
        GrB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
            AT, d, GrB_NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        GrB_TRY(LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            // printf("AT negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    (*pd_output) = d;
    d = NULL;
    LG_FREE_ALL;
    return (GrB_SUCCESS) ;
}


GrB_Info LAGraph_BF_basic_mxv
(
    GrB_Vector *pd_output,      //the pointer to the vector of distance
    const GrB_Matrix AT,        //transposed adjacency matrix for the graph
    const GrB_Index s           //given index of the source
)
{
    GrB_Info info;
    char *msg = NULL ;
    GrB_Index nrows, ncols;
    // tmp vector to store distance vector after n loops
    GrB_Vector d = NULL, dtmp = NULL;

    //LG_ASSERT (AT != NULL && pd_output != NULL, GrB_NULL_POINTER) ;

    *pd_output = NULL;
    GrB_TRY (GrB_Matrix_nrows (&nrows, AT)) ;
    GrB_TRY (GrB_Matrix_ncols (&ncols, AT)) ;
    //LG_ASSERT_MSG (nrows == ncols, -1002, "AT must be square") ;
    GrB_Index n = nrows;           // n = # of vertices in graph
    //LG_ASSERT_MSG (s < n, GrB_INVALID_INDEX, "invalid source node") ;

    // Initialize distance vector, change the d[s] to 0
    GrB_TRY (GrB_Vector_new(&d, GrB_FP64, n));
    GrB_TRY (GrB_Vector_setElement_FP64(d, 0, s));

    // copy d to dtmp in order to create a same size of vector
    GrB_TRY (GrB_Vector_dup(&dtmp, d));

    int64_t iter = 0;      //number of iterations
    bool same = false;     //variable indicating if d == dtmp

    // terminate when no new path is found or more than n-1 loops
    while (!same && iter < 20)
    {
        GrB_Index loc_nvals = 0;
        GrB_Vector_nvals (&loc_nvals, d);
        printf("nvals = %lld\n", loc_nvals);
        
        // excute semiring on d and AT, and save the result to d
        SAVE_STATS((GrB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
           AT, d, GrB_NULL))), "mxv_sssp", (sizeof(double)*2 + sizeof(size_t)), 1, AT); 
        
        //GrB_Index loc_nvals = 0;
        //GrB_Vector_nvals (&loc_nvals, d);
        //printf("nvals = %lld\n", loc_nvals);
        //SAVE_STATS(( GrB_TRY (GrB_vxm(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64, d, AT,
        //    GrB_NULL))), "vxm_sssp", (sizeof(double)*2 + sizeof(size_t)), 1, AT);

        //SAVE_STATS((GrB_TRY (GrB_mxv(dtmp, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
        //    AT, d, GrB_NULL))), "mxv_sssp", (sizeof(float)*2 + sizeof(size_t)), 1, AT);
        LAGraph_Vector_IsEqual (&same, dtmp, d, NULL);
        if (!same)
        {
            GrB_Vector ttmp = dtmp;
            dtmp = d;
            d = ttmp;
        }
        printf(same ? "true\n" : "false\n");
        iter++;
    }

    // check for negative-weight cycle only when there was a new path in the
    // last loop, otherwise, there can't be a negative-weight cycle.
    if (false)
    {
        // excute semiring again to check for negative-weight cycle
        GrB_TRY (GrB_mxv(dtmp, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP64,
            AT, d, GrB_NULL));

        // if d != dtmp, then there is a negative-weight cycle in the graph
        GrB_TRY(LAGraph_Vector_IsEqual (&same, dtmp, d, NULL));
        if (!same)
        {
            // printf("AT negative-weight cycle found. \n");
            LG_FREE_ALL;
            return (GrB_NO_VALUE) ;
        }
    }

    (*pd_output) = d;
    d = NULL;
    LG_FREE_ALL;
    return (GrB_SUCCESS) ;
}


int main (int argc, char **argv)
{
    printf("DOING shortest paths test\n");
    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;

    GrB_Matrix A = NULL ;
    GrB_Matrix Abool = NULL ;
    GrB_Vector PR = NULL ;
    FILE *f = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    int nt = NTHREAD_LIST ;
    int Nthreads [20] = { 0, THREAD_LIST } ;
    int nthreads_max ;
    LAGraph_TRY (LAGraph_GetNumThreads (&nthreads_max, NULL)) ;
    if (Nthreads [1] == 0)
    {
        // create thread list automatically
        Nthreads [1] = nthreads_max ;
        for (int t = 2 ; t <= nt ; t++)
        {
            Nthreads [t] = Nthreads [t-1] / 2 ;
            if (Nthreads [t] == 0) nt = t-1 ;
        }
    }
    printf ("threads to test: ") ;
    for (int t = 1 ; t <= nt ; t++)
    {
        int nthreads = Nthreads [t] ;
        if (nthreads > nthreads_max) continue ;
        printf (" %d", nthreads) ;
    }
    printf ("\n") ;

    double tic [2] ;

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    if (readproblem (&G, NULL,
        false, false, true, NULL, false, argc, argv) != 0) ERROR ;
    GrB_Index n, nvals ;
    GrB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GrB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    // determine the row degree property
    LAGraph_TRY (LAGraph_Property_RowDegree (G, msg)) ;

    //--------------------------------------------------------------------------
    // compute the pagerank
    //--------------------------------------------------------------------------

    // the GAP benchmark requires 16 trials
    int ntrials = 16 ;
    ntrials = 1 ;    // HACK to run just one trial
    printf ("# of trials: %d\n", ntrials) ;

    float damping = 0.85 ;
    float tol = 1e-4 ;
    int iters = 0, itermax = 100 ;

    for (int kk = 1 ; kk <= nt ; kk++)
    {
        int nthreads = Nthreads [kk] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, msg)) ;
        printf ("\n--------------------------- nthreads: %2d\n", nthreads) ;

        double total_time = 0 ;

        for (int trial = 0 ; trial < ntrials ; trial++)
        {
            GrB_free (&PR) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            //SAVE_STATS(LAGraph_TRY (LAGraph_VertexCentrality_PageRankGAP (&PR, G,
            //    damping, tol, itermax, &iters, msg)), "Page_Rank_LA", (sizeof(float)*2 + sizeof(size_t)), iters, (G->AT));
            //t1 ;
            SAVE_STATS( LAGraph_BF_basic_mxv(&PR, G->A, rand()%n - 1), "SSSP_LA_BF", (sizeof(double)*2 + sizeof(size_t)), iters, (G->A));
            LAGraph_TRY (LAGraph_Toc (&t1, tic, NULL)) ;
            printf ("trial: %2d time: %10.4f sec\n", trial, t1) ;
            total_time += t1 ;
        }

        double t = total_time / ntrials ;
        printf ("3f:%3d: avg time: %10.3f (sec), "
                "rate: %10.3f iters: %d\n", nthreads,
                t, 1e-6*((double) nvals) * iters / t, iters) ;
        fprintf (stderr, "Avg: PR (3f)      %3d: %10.3f sec: %s\n",
             nthreads, t, matrix_name) ;

    }

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    // TODO: check result from PageRank

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LAGraph_FREE_OUTER ;
    LAGraph_TRY (LAGraph_Finalize (msg)) ;
    return (0) ;
}
