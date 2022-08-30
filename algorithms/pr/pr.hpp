/*
 * This file uses algorithm implementation from LAGraph, which is available under
 * their custom license. For details, see https://github.com/GraphBLAS/LAGraph/blob/reorg/LICENSE
 * */

/**
  @file pr.hpp
  @author S.krymskiy
  @version Revision 1.1
  @brief PR algorithm.
  @details This file uses algorithm implementation from LAGraph, which is available under
  their custom license. For details, see https://github.com/GraphBLAS/LAGraph/blob/reorg/LICENSE.
  @date May 12, 2022
*/

#pragma once

#define GrB_Matrix lablas::Matrix<float>*
#define GrB_Vector lablas::Vector<float>*
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

//! Lablas namespace

namespace lablas {

//! Algorithm namespace

namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * LAGraph_page_rank_sinks function.
 * @brief The function does...
 * @param centrality centrality(i): GAP-style pagerank of node i
 * @param G input graph
 * @param iters output: number of iterations taken
 * @param itermax maximum number of iterations (typically 100)
 * @param damping damping factor (typically 0.85)
 * @param tol  stopping tolerance (typically 1e-4)
*/

void LAGraph_page_rank_sinks (GrB_Vector centrality, // centrality(i): GAP-style pagerank of node i
        // inputs:
                              LAGraph_Graph<float> *G,        // input graph
                              int *iters,                     // output: number of iterations taken
                              int itermax = 100,              // maximum number of iterations (typically 100)
                              double damping = 0.85,           // damping factor (typically 0.85)
                              double tol = 1e-4               // stopping tolerance (typically 1e-4) ;
)
{
    double loop_vector_clear_time = 0;
    double loop_assign_time = 0;
    double loop_reduce_time = 0;
    double loop_ewisemult_time = 0;
    double loop_mxv_time = 0;
    double loop_apply_time = 0;

    double t_op_start;

    const auto t1 = omp_get_wtime();
    GrB_Matrix AT = G->AT;
    lablas::Vector<Index>* d_out = G->coldegree; // TODO row degree and transpose
    GrB_Vector r = NULL, *d = NULL, *t = NULL, *w = NULL, *d1 = NULL ;
    lablas::Descriptor desc;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GrB_TRY (GrB_Matrix_nrows (&n, AT)) ;

    const double scaled_damping = (1 - damping) / n ;
    float rdiff = 1 ;       // first iteration is always done

    const auto t2 = omp_get_wtime();

    // r = 1 / n
    GrB_TRY (GrB_Vector_new (&t, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&r, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&w, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    t->set_name("t");
    r->set_name("r");
    w->set_name("w");
    d->set_name("d");
    d1->set_name("d1");

    const auto t3 = omp_get_wtime();

    double init_rank = 1.0/n;
    t_op_start = omp_get_wtime();
    GrB_TRY (GrB_assign (r, MASK_NULL, NULL, init_rank, GrB_ALL, n, NULL)) ;
    loop_assign_time += omp_get_wtime() - t_op_start;

    // find all sinks, where sink(i) = true if node i has d_out(i)=0, or with
    // d_out(i) not present.  LAGraph_Property_RowDegree computes d_out =
    // G->rowdegree so that it has no explicit zeros, so a structural mask can
    // be used here.
    lablas::Vector<bool>* sink = NULL;
    lablas::Vector<float>* rsink = NULL;
    GrB_Index nsinks, nvals ;
    GrB_TRY (GrB_Vector_nvals (&nvals, d_out)) ;
    nsinks = n - nvals ;
    if (nsinks > 0)
    {
        // sink<!struct(d_out)> = true
        GrB_TRY (GrB_Vector_new (&sink, GrB_BOOL, n)) ;
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_assign (sink, d_out, NULL, (bool)true, GrB_ALL, n, GrB_DESC_SC)) ;
        loop_assign_time += omp_get_wtime() - t_op_start;
        GrB_TRY (GrB_Vector_new (&rsink, GrB_FP32, n));
        sink->set_name("sink");
        rsink->set_name("rsink");
    }

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    t_op_start = omp_get_wtime();
    GrB_TRY (GrB_apply (d, MASK_NULL, NULL, GrB_DIV_FP32, d_out, damping, &desc)) ;
    loop_apply_time += omp_get_wtime() - t_op_start;

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    t_op_start = omp_get_wtime();
    GrB_TRY (GrB_assign (d1, MASK_NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    loop_assign_time += omp_get_wtime() - t_op_start;

    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, MASK_NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;

    const auto t4 = omp_get_wtime();

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------
    for ((*iters) = 0 ; (*iters) < itermax; (*iters)++)
    {
        float teleport = scaled_damping ; // teleport = (1 - damping) / n
        if (nsinks > 0)
        {
            const float damping_over_n = damping / n ;
            // handle the sinks: teleport += (damping/n) * sum (r (sink))
            // rsink<struct(sink)> = r
            t_op_start = omp_get_wtime();
            GrB_TRY (GrB_Vector_clear (rsink)) ;
            loop_vector_clear_time += omp_get_wtime() - t_op_start;

            t_op_start = omp_get_wtime();
            GrB_TRY (GrB_assign (rsink, sink, NULL, r, GrB_ALL, n, GrB_DESC_S));
            loop_assign_time += omp_get_wtime() - t_op_start;

            // sum_rsink = sum (rsink)
            float sum_rsink = 0 ;
            t_op_start = omp_get_wtime();
            GrB_TRY (GrB_reduce (&sum_rsink, NULL, GrB_PLUS_MONOID_FP32, rsink, NULL)) ;
            loop_reduce_time += omp_get_wtime() - t_op_start;
            teleport += damping_over_n * sum_rsink ;
        }

        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d

        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_eWiseMult (w, MASK_NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        loop_ewisemult_time += omp_get_wtime() - t_op_start;

        // r = teleport
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_assign (r, MASK_NULL, NULL, teleport, GrB_ALL, n, NULL)) ;
        loop_assign_time += omp_get_wtime() - t_op_start;

        // r += A'*w
        t_op_start = omp_get_wtime();
        SAVE_STATS((GrB_TRY (GrB_mxv (r, MASK_NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32, AT, w, &desc))),
                   "pr_mxv", (sizeof(float)*2 + sizeof(size_t)), 1, AT);
        loop_mxv_time += omp_get_wtime() - t_op_start;

        // t -= r
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_assign (t, MASK_NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        loop_assign_time += omp_get_wtime() - t_op_start;
        // t = abs (t)
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_apply (t, MASK_NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        loop_apply_time += omp_get_wtime() - t_op_start;
        // rdiff = sum (t)
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;
        loop_reduce_time += omp_get_wtime() - t_op_start;

        //float ranks_sum = 0;
        double ranks_sum = 0;
        t_op_start = omp_get_wtime();
        GrB_TRY (GrB_reduce (&ranks_sum, NULL, GrB_PLUS_MONOID_FP32, r, NULL));
        loop_reduce_time += omp_get_wtime() - t_op_start;
#ifdef __DEBUG_INFO__
        cout << "ranks sum: " << ranks_sum << endl;
#endif
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    const auto t5 = omp_get_wtime();

    centrality->dup(r);
    GrB_free (&d1) ;
    GrB_free (&d);
    GrB_free (&t);
    GrB_free (&w);
    GrB_free (&r);
    if (nsinks > 0)
    {
        GrB_free (&sink);
        GrB_free (&rsink);
    }
    const auto t6 = omp_get_wtime();

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_before_allocation", (t2 - t1) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_vector_allocation", (t3 - t2) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_before_loop", (t4 - t3) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_loop", (t5 - t4) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_free", (t6 - t5) * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_vector_clear", loop_vector_clear_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_assign", loop_assign_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_reduce", loop_reduce_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_ewisemult", loop_ewisemult_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_mxv_total_time", loop_mxv_time * 1000, 0.0, 0.0, 0ll);
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "pr_apply", loop_apply_time * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
}

#undef GrB_Matrix
#undef GrB_Vector
#undef MASK_NULL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Алгоритм преобразования LAGraph кода графового алгоритма к коду, совместимому с GB_Kun
// удалить инициализуию LAgraph
// lablas::Vector<Index>* d_out
// *
// изменить маски NULL на MASK_TEMP
// изменить дескриптор
