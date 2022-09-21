//------------------------------------------------------------------------------
// LAGraph/Test2/ConnectedComponents/test_cc: test LAGraph_ConnectedComponents
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

// Contributed by Tim Davis, Texas A&M

// Usage: test_cc can be used with both stdin or a file as its input,
// in either grb or mtx format.

//------------------------------------------------------------------------------

#include "LAGraph_demo.h"
#include "LAGraphX.h"
#include "LG_alg_internal.h"

#undef  LAGraph_FREE_ALL
#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&components) ;        \
    GrB_free (&components2) ;       \
}

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

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

GrB_Index countCC (GrB_Vector f, GrB_Index n)
{
    GrB_Index nCC = 0;
    GrB_Index *w_val = (GrB_Index *) LAGraph_Malloc (n, sizeof (GrB_Index)) ;
    if (w_val == NULL) { printf ("out of memory\n") ; abort ( ) ; }
    #if LG_SUITESPARSE
    // SuiteSparse:GraphBLAS allows NULL inputs to GrB_Vector_extractTuples
    GrB_Index *i_val = NULL ;
    #else
    GrB_Index *i_val = (GrB_Index *) LAGraph_Malloc (n, sizeof (GrB_Index)) ;
    if (i_val == NULL) { printf ("out of memory\n") ; abort ( ) ; }
    #endif
    GrB_Vector_extractTuples (i_val, w_val, &n, f) ;
    for (GrB_Index i = 0; i < n; i++)
    {
        if (w_val[i] == i)
        {
            nCC++ ;
        }
    }
    LAGraph_Free ((void **) &i_val) ;
    LAGraph_Free ((void **) &w_val) ;
    return nCC;
}

int main (int argc, char **argv)
{

    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;
    GrB_Vector components = NULL, components2 = NULL ;

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

    double tt [nthreads_max+1] ;

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    fprintf (stderr, "\n%s:\n", matrix_name) ;
    if (readproblem (&G,
        NULL,   // no source nodes
        true,   // make the graph undirected, and symmetrize the matrix
        false,  // do not remove self-edges
        true,   // structural only, no values needed
        NULL,   // no type preference
        false,  // do not ensure all entries positive
        argc, argv) != 0) ERROR ;
    GrB_Index n, nvals ;
    GrB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GrB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    //--------------------------------------------------------------------------
    // begin tests
    //--------------------------------------------------------------------------

    double tic [2], tcheck ;

    // warmup
    LAGraph_TRY (LAGraph_ConnectedComponents (&components, G, msg)) ;
    GrB_Index nCC = countCC (components, n) ;
    printf ("nCC: %20.0g\n", (double) nCC) ;

#if 0 & LG_CHECK_RESULT
    LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
    int result = LG_check_cc (components, G, msg) ;
    if (result != 0)
    {
        printf ("test failure: (%d) %s\n", result, msg) ;
    }
    LAGraph_TRY (LAGraph_Toc (&tcheck, tic, NULL)) ;
    LAGraph_TRY (result) ;
    printf ("LG_check_cc passed, time: %g\n", tcheck) ;
#endif

    #define NTRIALS 16
    // #define NTRIALS 1
    printf ("# of trials: %d\n\n", NTRIALS) ;

    //--------------------------------------------------------------------------
    // LAGraph_ConnectedComponents
    //--------------------------------------------------------------------------

    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            double ttrial ;
            GrB_free (&components2) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            LAGraph_TRY (LAGraph_ConnectedComponents (&components2, G, msg)) ;
            LAGraph_TRY (LAGraph_Toc (&ttrial, tic, NULL)) ;
            ttt += ttrial ;
            printf ("SV6:      nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV6:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV6:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }

    //--------------------------------------------------------------------------
    // 7: draft version
    //--------------------------------------------------------------------------

#if 0
    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            double ttrial ;
            GrB_free (&components2) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            LAGraph_TRY (LG_CC_7 (&components2, G, msg)) ;
            LAGraph_TRY (LAGraph_Toc (&ttrial, tic, NULL)) ;
            ttt += ttrial ;
            printf ("SV7:      nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV7:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV7:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }
#endif

    //--------------------------------------------------------------------------
    // LG_CC_FastSV5: using 32-bit integers
    //--------------------------------------------------------------------------

    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            double ttrial ;
            GrB_free (&components2) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            LAGraph_TRY (LG_CC_FastSV5 (&components2, G, msg)) ;
            LAGraph_TRY (LAGraph_Toc (&ttrial, tic, NULL)) ;
            ttt += ttrial ;
            printf ("SV5b:     nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV5b:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV5b:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }

    //--------------------------------------------------------------------------
    // LG_CC_Boruvka
    //--------------------------------------------------------------------------

    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = 1 /* NTRIALS */ ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            double ttrial ;
            GrB_free (&components2) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            SAVE_TEPS(LAGraph_TRY (LG_CC_Boruvka (&components2, G, msg)), "CC_Boruvka", 1, (G->A)) ;
            LAGraph_TRY (LAGraph_Toc (&ttrial, tic, NULL)) ;
            ttt += ttrial ;
            printf ("Boruvka:  nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("Boruvka:  nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "Boruvka:  nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }

    //--------------------------------------------------------------------------
    // LAGraph_cc_lacc
    //--------------------------------------------------------------------------

    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = 1 /* NTRIALS */ ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            double ttrial ;
            GrB_free (&components2) ;
            LAGraph_TRY (LAGraph_Tic (tic, NULL)) ;
            SAVE_TEPS(( LAGraph_TRY (LAGraph_cc_lacc (&components2, G->A, false)) ), "cc_lacc", 1, (G->A));
            LAGraph_TRY (LAGraph_Toc (&ttrial, tic, NULL)) ;
            ttt += ttrial ;
            printf ("LACC:     nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("LACC:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "LACC:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LAGraph_FREE_ALL ;
    LAGraph_TRY (LAGraph_Finalize (msg)) ;
    return (0) ;
}
