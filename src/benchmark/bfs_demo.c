//------------------------------------------------------------------------------
// LAGraph/Test2/BreadthFirstSearch/test_bfs.c: test LAGraph_BreadthFirstSearch
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause

//------------------------------------------------------------------------------

#include "LAGraph_demo.h"

#define NTHREAD_LIST 1
#define THREAD_LIST 0

// #define NTHREAD_LIST 8
// #define THREAD_LIST 8, 7, 6, 5, 4, 3, 2, 1

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_Delete (&G, msg) ;      \
    GrB_free (&A) ;                 \
    GrB_free (&Abool) ;             \
    GrB_free (&parent) ;            \
    GrB_free (&level) ;             \
    GrB_free (&SourceNodes) ;       \
}

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
    GrB_Vector level = NULL ;
    GrB_Vector parent = NULL ;
    GrB_Matrix SourceNodes = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    uint64_t seed = 1 ;
    FILE *f ;
    int nthreads ;

    int nt = NTHREAD_LIST ;
    int Nthreads [20] = { 0, THREAD_LIST } ;
    int nthreads_max ;
    LAGraph_TRY (LAGraph_GetNumThreads (&nthreads_max, NULL)) ;
    printf ("nthreads_max: %d\n", nthreads_max) ;
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

    double tpl [nthreads_max+1][2] ;
    double tp [nthreads_max+1][2] ;
    double tl [nthreads_max+1][2] ;

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    if (readproblem (&G, &SourceNodes,
        false, false, true, NULL, false, argc, argv) != 0) ERROR ;

    // compute G->rowdegree
    LAGraph_TRY (LAGraph_Property_RowDegree (G, msg)) ;

    // compute G->coldegree, just to test it (not needed for any tests)
    LAGraph_TRY (LAGraph_Property_ColDegree (G, msg)) ;

    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, G->A)) ;

    //--------------------------------------------------------------------------
    // get the source nodes
    //--------------------------------------------------------------------------

    GrB_Index ntrials ;
    GrB_TRY (GrB_Matrix_nrows (&ntrials, SourceNodes)) ;

    // HACK
    // ntrials = 4 ;

    //--------------------------------------------------------------------------
    // warmup
    //--------------------------------------------------------------------------

    int64_t src ;
    double twarmup, tw [2] ;
    GrB_TRY (GrB_Matrix_extractElement (&src, SourceNodes, 0, 0)) ;
    LAGraph_TRY (LAGraph_Tic (tw, msg)) ;
    LAGraph_TRY (LAGraph_BreadthFirstSearch (NULL, &parent,
        G, src, false, msg)) ;
    GrB_free (&parent) ;
    LAGraph_TRY (LAGraph_Toc (&twarmup, tw, msg)) ;
    printf ("warmup: parent only, pushonly: %g sec\n", twarmup) ;

    //--------------------------------------------------------------------------
    // run the BFS on all source nodes
    //--------------------------------------------------------------------------
    GrB_Vector row_degs = G->rowdegree ;
    GrB_Vector col_degs = G->coldegree ;
    for(int i = 0; i < 10; i++)
    {
        GrB_Index val = 0;
        GrB_Vector_extractElement(&val, row_degs, i);
        int row_val = val;
        GrB_Vector_extractElement(&val, col_degs, i);
        int col_val = val;
        printf("row deg = %d col deg = %d \n", row_val, col_val);
    }

    for (int tt = 1 ; tt <= nt ; tt++)
    {
        int nthreads = Nthreads [tt] ;
        if (nthreads > nthreads_max) continue ;
        LAGraph_TRY (LAGraph_SetNumThreads (nthreads, msg)) ;

        tp [nthreads][0] = 0 ;
        tl [nthreads][0] = 0 ;
        tpl [nthreads][0] = 0 ;

        tp [nthreads][1] = 0 ;
        tl [nthreads][1] = 0 ;
        tpl [nthreads][1] = 0 ;

        printf ("\n------------------------------- threads: %2d\n", nthreads) ;
        for (int trial = 0 ; trial < 1000/*ntrials*/ ; trial++)
        {
            int64_t src ;
            // src = SourceNodes [trial]
            GrB_TRY (GrB_Matrix_extractElement (&src, SourceNodes, trial, 0)) ;
            src-- ; // convert from 1-based to 0-based
            double tcheck, ttrial, tic [2] ;
            
            /*while(true)
            {
                GrB_Index try = rand() % 1000;
                GrB_Index val = 0;
                GrB_Vector_extractElement(&val, row_degs, try);
                int row_val = val;
                GrB_Vector_extractElement(&val, col_degs, try);
                int col_val = val;
                //printf("row deg = %d col deg = %d \n", row_val, col_val);
                if(row_val == 0 || col_val == 0)
                    continue;
                src = try;
                printf("selected source %lld\n", src);
                break;
            }*/
            src = trial;

            for (int pp = 0 ; pp <= 1 ; pp++)
            {

                bool pushpull = (pp == 1) ;

                //--------------------------------------------------------------
                // BFS to compute just parent
                //--------------------------------------------------------------

                GrB_free (&parent) ;
                LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                LAGraph_TRY (LAGraph_BreadthFirstSearch (NULL, &parent,
                    G, src, pushpull, msg)) ;
                LAGraph_TRY (LAGraph_Toc (&ttrial, tic, msg)) ;
                tp [nthreads][pp] += ttrial ;
                printf ("parent only  %s trial: %2d threads: %2d "
                    "src: %g %10.4f sec\n",
                    (pp == 0) ? "pushonly" : "pushpull",
                    trial, nthreads, (double) src, ttrial) ;
                fflush (stdout) ;

                int32_t maxlevel ;
                GrB_Index nvisited ;

#if LG_CHECK_RESULT
                // check the result (this is very slow so only do it for one trial)
                if (trial == 0)
                {
                    LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                    LAGraph_TRY (LG_check_bfs (NULL, parent, G, src, msg)) ;
                    LAGraph_TRY (LAGraph_Toc (&tcheck, tic, msg)) ;
                    printf ("    n: %g check: %g sec\n", (double) n, tcheck) ;
                }
#endif

                GrB_free (&parent) ;

                //--------------------------------------------------------------
                // BFS to compute just level
                //--------------------------------------------------------------

                GrB_free (&level) ;

                LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                SAVE_TEPS(LAGraph_TRY (LAGraph_BreadthFirstSearch (&level, NULL,
                    G, src, pushpull, msg)), "BFS_levels_only", 1, (G->A));
                LAGraph_TRY (LAGraph_Toc (&ttrial, tic, msg)) ;
                tl [nthreads][pp] += ttrial ;

                GrB_TRY (GrB_reduce (&maxlevel, NULL, GrB_MAX_MONOID_INT32,
                    level, NULL)) ;
                printf ("level only   %s trial: %2d threads: %2d "
                    "src: %g %10.4f sec maxlevel: %d\n",
                    (pp == 0) ? "pushonly" : "pushpull",
                    trial, nthreads, (double) src, ttrial, maxlevel) ;
                fflush (stdout) ;

#if LG_CHECK_RESULT
                // check the result (this is very slow so only do it for one trial)
                if (trial == 0)
                {
                    LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                    LAGraph_TRY (LG_check_bfs (level, NULL, G, src, msg)) ;
                    GrB_TRY (GrB_Vector_nvals (&nvisited, level)) ;
                    LAGraph_TRY (LAGraph_Toc (&tcheck, tic, msg)) ;
                    printf ("    n: %g max level: %d nvisited: %g "
                        "check: %g sec\n", (double) n, maxlevel,
                        (double) nvisited, tcheck) ;
                }
#endif

                GrB_free (&level) ;

                //--------------------------------------------------------------
                // BFS to compute both parent and level
                //--------------------------------------------------------------

                GrB_free (&parent) ;
                GrB_free (&level) ;
                LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                LAGraph_TRY (LAGraph_BreadthFirstSearch (&level, &parent,
                    G, src, pushpull, msg)) ;
                LAGraph_TRY (LAGraph_Toc (&ttrial, tic, msg)) ;
                tpl [nthreads][pp] += ttrial ;

                GrB_TRY (GrB_reduce (&maxlevel, NULL, GrB_MAX_MONOID_INT32,
                    level, NULL)) ;
                printf ("parent+level %s trial: %2d threads: %2d "
                    "src: %g %10.4f sec\n",
                    (pp == 0) ? "pushonly" : "pushpull",
                    trial, nthreads, (double) src, ttrial) ;
                fflush (stdout) ;

#if LG_CHECK_RESULT
                // check the result (this is very slow so only do it for one trial)
                if (trial == 0)
                {
                    LAGraph_TRY (LAGraph_Tic (tic, msg)) ;
                    LAGraph_TRY (LG_check_bfs (level, parent, G, src, msg)) ;
                    GrB_TRY (GrB_Vector_nvals (&nvisited, level)) ;
                    LAGraph_TRY (LAGraph_Toc (&tcheck, tic, msg)) ;
                    printf ("    n: %g max level: %d nvisited: %g "
                        "check: %g sec\n",
                        (double) n, maxlevel, (double) nvisited, tcheck) ;
                }
#endif

                GrB_free (&parent) ;
                GrB_free (&level) ;
            }
        }

        for (int pp = 0 ; pp <= 1 ; pp++)
        {
            tp  [nthreads][pp] = tp  [nthreads][pp] / ntrials ;
            tl  [nthreads][pp] = tl  [nthreads][pp] / ntrials ;
            tpl [nthreads][pp] = tpl [nthreads][pp] / ntrials ;

            fprintf (stderr, "Avg: BFS %s parent only  threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tp [nthreads][pp], matrix_name) ;
#if 1
            fprintf (stderr, "Avg: BFS %s level only   threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tl [nthreads][pp], matrix_name) ;

            fprintf (stderr, "Avg: BFS %s level+parent threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tpl [nthreads][pp], matrix_name) ;
#endif

            printf ("Avg: BFS %s parent only  threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tp [nthreads][pp], matrix_name) ;

#if 1
            printf ("Avg: BFS %s level only   threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tl [nthreads][pp], matrix_name) ;

            printf ("Avg: BFS %s level+parent threads %3d: "
                "%10.3f sec: %s\n",
                 (pp == 0) ? "pushonly" : "pushpull",
                 nthreads, tpl [nthreads][pp], matrix_name) ;
#endif
        }
    }
    // restore default
    LAGraph_TRY (LAGraph_SetNumThreads (nthreads_max, msg)) ;
    printf ("\n") ;

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LAGraph_FREE_ALL ;
    LAGraph_TRY (LAGraph_Finalize (msg)) ;
    return (0) ;
}
