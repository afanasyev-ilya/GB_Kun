#pragma once

#define GrB_Matrix lablas::Matrix<float>*
#define GrB_Vector lablas::Vector<float>*
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

int LAGraph_VertexCentrality_PageRankGAP (GrB_Vector* centrality, // centrality(i): GAP-style pagerank of node i
                                          // inputs:
                                          LAGraph_Graph<float> *G,        // input graph
                                          int *iters,                     // output: number of iterations taken
                                          float damping = 0.85,           // damping factor (typically 0.85)
                                          float tol = 1e-4,               // stopping tolerance (typically 1e-4) ;
                                          int itermax = 100)              // maximum number of iterations (typically 100)
{
    GrB_Matrix AT = G->AT;
    lablas::Vector<Index>* d_out = G->rowdegree ;
    GrB_Vector r = NULL, *d = NULL, *t = NULL, *w = NULL, *d1 = NULL ;
    lablas::Descriptor desc;

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
    GrB_TRY (GrB_assign (r, MASK_NULL, NULL, 1.0 / n, GrB_ALL, n, NULL)) ;

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GrB_TRY (GrB_apply (d, MASK_NULL, NULL, GrB_DIV_FP32, d_out, damping, &desc)) ;

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, MASK_NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, MASK_NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    for ((*iters) = 0 ; (*iters) < itermax && rdiff > tol ; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GrB_TRY (GrB_eWiseMult (w, MASK_NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, MASK_NULL, NULL, teleport, GrB_ALL, n, NULL)) ;

        // r += A'*w
        double t1, t2;
        GrB_Index nvals = 0;
        if(true)
        {
            GrB_Matrix_nvals(&nvals, AT);
            printf("matrix has %ld\n edges", nvals);
            t1 = omp_get_wtime();
        }
        GrB_TRY (GrB_mxv (r, MASK_NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32, AT, w, &desc)) ;
        if(true)
        {
            t2 = omp_get_wtime();
            double gflop = nvals * 2.0 / ((t2 - t1)*1e9);
            printf("edges: %lf\n", nvals);
            printf("SPMV time %lf (ms)\n", (t2-t1)*1000);
            printf("SPMV perf %lf (GFLop/s)\n", gflop);
            printf("SPMV BW %lf (GB/s)\n", nvals * (sizeof(float)*2 + sizeof(size_t))/((t2 - t1)*1e9));
        }

        // t -= r
        GrB_TRY (GrB_assign (t, MASK_NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GrB_TRY (GrB_apply (t, MASK_NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;

        float ranks_sum = 0;
        GrB_TRY (GrB_reduce (&ranks_sum, NULL, GrB_PLUS_MONOID_FP32, r, NULL));
        cout << "ranks sum : " << ranks_sum << endl;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*centrality) = r ;
    GrB_free (&d1) ;
    GrB_free (&d);
    GrB_free (&t);
    GrB_free (&w);
    return 0;
}

#undef GrB_Matrix
#undef GrB_Vector
#undef MASK_NULL

// удалить инициализуию LAgraph
// lablas::Vector<Index>* d_out
// *
// изменить маски NULL на MASK_TEMP
// изменить дескриптор