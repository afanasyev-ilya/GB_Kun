#pragma once

#define GrB_Matrix lablas::Matrix<float>*
#define GrB_Vector lablas::Vector<float>*
#define TEMP_NULL static_cast<const lablas::Vector<float>*>(NULL)

int LAGraph_VertexCentrality_PageRankGAP // returns -1 on failure, 0 on success
        (
                // outputs:
                GrB_Vector* centrality, // centrality(i): GAP-style pagerank of node i
                // inputs:
                LAGraph_Graph<float> *G,        // input graph
                int *iters,                      // output: number of iterations taken
                float damping = 0.85,           // damping factor (typically 0.85)
                float tol = 1e-4,               // stopping tolerance (typically 1e-4) ;
                int itermax = 100              // maximum number of iterations (typically 100)
        )
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector r = NULL;
    GrB_Vector d = NULL;
    GrB_Vector t = NULL;
    GrB_Vector w = NULL;
    GrB_Vector d1 = NULL;

    lablas::Descriptor desc;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Matrix AT = G->AT;
    lablas::Vector<GrB_Index>* d_out = G->rowdegree;

    GrB_Index n ;
    //(*centrality) = NULL ; // TODO
    GrB_TRY(GrB_Matrix_nrows (&n, AT)) ;

    const float teleport = (1 - damping) / n ;
    float rdiff = 1 ;       // first iteration is always done

    // r = 1 / n
    GrB_TRY (GrB_Vector_new (&t, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&r, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&w, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (r, TEMP_NULL, NULL, 1.0 / n, GrB_ALL, n, NULL)) ;

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GrB_TRY (GrB_apply (d, TEMP_NULL, NULL, GrB_DIV_FP32, d_out, damping, NULL)) ;

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, TEMP_NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, TEMP_NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;
    GrB_free (&d1) ;

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    cout << rdiff << " " << tol << endl;
    for ((*iters) = 0 ; (*iters) < 1 && rdiff > tol; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GrB_TRY (GrB_eWiseMult (w, TEMP_NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, TEMP_NULL, NULL, teleport, GrB_ALL, n, NULL)) ;
        // r += A'*w
        GrB_TRY (GrB_mxv (r, TEMP_NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32, AT, w, &desc)) ;
        // t -= r
        GrB_TRY (GrB_assign (t, TEMP_NULL, NULL, r, GrB_ALL, n, NULL)) ;

        // t = abs (t)
        GrB_TRY (GrB_apply (t, TEMP_NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL));
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    float ranks_sum = 0;
    GrB_TRY (GrB_reduce (&ranks_sum, NULL, GrB_PLUS_MONOID_FP32, r, NULL));
    r->print();
    cout << "ranks sum : " << ranks_sum << endl;

    (*centrality) = r ;
    return 0;
}

#undef GrB_Matrix
#undef GrB_Vector
#undef TEMP_NULL
