#pragma once

int LAGraph_VertexCentrality_PageRankGAP // returns -1 on failure, 0 on success
        (
                // outputs:
                //GrB_Vector *centrality, // centrality(i): GAP-style pagerank of node i
                // inputs:
                LAGraph_Graph<float> *G,        // input graph
                float damping,          // damping factor (typically 0.85)
                float tol,              // stopping tolerance (typically 1e-4) ;
                int itermax,            // maximum number of iterations (typically 100)
                int *iters             // output: number of iterations taken
                //char *msg
        )
{

    #define GrB_Matrix lablas::Matrix<float>*
    #define GrB_Vector lablas::Vector<float>*
    #define TEMP_NULL static_cast<const lablas::Vector<float>*>(NULL)
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector r = NULL;
    GrB_Vector d = NULL;
    GrB_Vector t = NULL;
    GrB_Vector w = NULL;
    GrB_Vector d1 = NULL;

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
    GrB_TRY (GrB_assign (r, TEMP_NULL, nullptr, 1.0 / n, GrB_ALL, n, NULL)) ;

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GrB_TRY (GrB_apply (d, TEMP_NULL, nullptr, GrB_DIV_FP32, d_out, damping, NULL)) ;
    d_out->print();
    d->print();

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, TEMP_NULL, nullptr, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    d1->print();
    d->print();
    GrB_TRY (GrB_eWiseAdd (d, TEMP_NULL, nullptr, GrB_MAX_FP32, d1, d, NULL)) ;
    GrB_free (&d1) ;
    d->print();

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    for ((*iters) = 0 ; (*iters) < itermax && rdiff > tol ; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        /*GrB_TRY (GrB_eWiseMult (w, NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, NULL, NULL, teleport, GrB_ALL, n, NULL)) ;
        // r += A'*w
        GrB_TRY (GrB_mxv (r, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
            AT, w, NULL)) ;
        // t -= r
        GrB_TRY (GrB_assign (t, NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GrB_TRY (GrB_apply (t, NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;*/
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    /*(*centrality) = r ;
    LAGraph_FREE_WORK ;
    return (0);*/

    #undef GrB_Matrix
    #undef GrB_Vector
    #undef TEMP_NULL
}

// вопросы для обсуждения

// GrB_Vector - это указатель на Vector?

// у нас есть указание шаблонных парамтеров vxm<float, float, float, float> -- как его избежать?
// делать отдельные варинаты функций под возможность NULL?

// где храним транспонированную матрицу? как у них, или нет?

// реализация интеграции
// 1. wrappers GrB_assign (t, NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL) - внутри вызов наших функций, совместимых с graphBLAST
// 2. define LAGraph_plus_second_fp32
// 3. структура LAGraph_Graph
// 4. макросы GrB_TRY / LG_CHECK
