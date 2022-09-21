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

#define LAGraph_FREE_ALL                        \
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

#define SAVE_TEPS(call_instruction, op_name, iterations, matrix)                        \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                  \
double my_perf = iterations*(my_nvals / ((my_t2 - my_t1)*1e6));                         \
double my_bw = 0;                                                                       \
FILE *my_f;                                                                             \
my_f = fopen("perf_stats.txt", "a");                                                    \
fprintf(my_f, "%s %lf (ms) %lf (MTEPS/s) %lf (GB/s) %ld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);                                                                           \

int main (int argc, char **argv)
{

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
    double t1_mes = omp_get_wtime();
    if (readproblem (&G, NULL,
        false, false, true, NULL, false, argc, argv) != 0) ERROR ;
    double t2_mes = omp_get_wtime();
    save_time_in_sec("whole_preprocess", t2_mes - t1_mes);
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
    // ntrials = 1 ;    // HACK to run just one trial
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
            t1_mes = omp_get_wtime();
            LAGraph_TRY (LAGraph_VertexCentrality_PageRankGAP (&PR, G,
                damping, tol, itermax, &iters, msg));
            GrB_Index mes_nvals = 0; 
            GrB_Matrix_nvals(&mes_nvals, G->AT);  
            t2_mes = omp_get_wtime();
            save_teps("page_rank", t2_mes - t1_mes, mes_nvals, iters);
            double t1 ;
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

    LAGraph_FREE_ALL ;
    LAGraph_TRY (LAGraph_Finalize (msg)) ;
    return (0) ;
}
