template<typename T>
int LAGraph_VertexCentrality_PageRankGAP // returns -1 on failure, 0 on success
        (
                // outputs:
                //GrB_Vector *centrality, // centrality(i): GAP-style pagerank of node i
                // inputs:
                LAGraph_Graph<T> *G,        // input graph
                float damping,          // damping factor (typically 0.85)
                float tol,              // stopping tolerance (typically 1e-4) ;
                int itermax            // maximum number of iterations (typically 100)
                //int *iters,             // output: number of iterations taken
                //char *msg
        )
{

    #define GrB_Matrix lablas::Matrix<float>*
    #define GrB_Vector lablas::Vector<float>*
    #define TEMP_NULL static_cast<const lablas::Vector*>(NULL)
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector r = NULL;
    GrB_Vector d = NULL;
    GrB_Vector t = NULL;
    GrB_Vector w = NULL;
    GrB_Vector d1 = NULL;
    /*LG_CHECK (centrality == NULL, -1, "centrality is NULL") ;
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
    LG_CHECK (d_out == NULL, -1, "G->rowdegree is required") ;*/

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Matrix AT = G->A;

    GrB_Index n ;
    //(*centrality) = NULL ; // TODO
    GrB_TRY(GrB_Matrix_nrows (&n, AT)) ;

    const float teleport = (1 - damping) / n ;
    float rdiff = 1 ;       // first iteration is always done

    // r = 1 / n
    GrB_TRY (GrB_Vector_new (&t, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&r, GrB_FP32, n)) ;
    GrB_TRY (GrB_Vector_new (&w, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (r, static_cast<const lablas::Vector<T>*>(NULL), nullptr, 1.0 / n, GrB_ALL, n, NULL)) ;
    r->print();

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GrB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    //GrB_TRY (GrB_apply (d, NULL, NULL, GrB_DIV_FP32, d_out, damping, NULL)) ;

    // d1 = 1 / damping
    /*float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;
    GrB_free (&d1) ;*/

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    /*for ((*iters) = 0 ; (*iters) < itermax && rdiff > tol ; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GrB_TRY (GrB_eWiseMult (w, NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, NULL, NULL, teleport, GrB_ALL, n, NULL)) ;

        GrB_Index nvals = 0;
        GrB_Matrix_nvals(&nvals, AT);
        printf("matrix has %d %ld\n edges", nvals, nvals);
        printf("hi Elijah\n");
        GrB_Index v_nvals = 0, v_size;
        GrB_Vector_nvals(&v_nvals, w);
        GrB_Vector_size(&v_size, w);
        printf("vector nvals: %ld / %ld\n", v_nvals, v_size);
        printf("vector nvals: %lf / %lf\n", (double)v_nvals, (double)v_size);
        // r += A'*w
        double t1 = omp_get_wtime();
        GrB_TRY (GrB_mxv (r, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
                          AT, w, NULL)) ;
        double t2 = omp_get_wtime();
        double gflop = nvals * 2.0 / ((t2 - t1)*1e9);
        double edges = nvals;
        printf("edges: %lf\n", edges);
        printf("SPMV time %lf (ms)\n", (t2-t1)*1000);
        printf("SPMV perf %lf (GFLop/s)\n", gflop);
        printf("SPMV BW %lf (GB/s)\n", nvals * (sizeof(float)*2 + sizeof(size_t))/((t2 - t1)*1e9));

        // t -= r
        GrB_TRY (GrB_assign (t, NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GrB_TRY (GrB_apply (t, NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*centrality) = r ;
    LAGraph_FREE_WORK ;*/
    return (0);

    #undef GrB_Matrix

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
