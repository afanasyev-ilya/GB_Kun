#pragma once

#define GrB_Matrix lablas::Matrix<float>*
#define GrB_Vector lablas::Vector<float>*
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

namespace lablas {
namespace algorithm {

void page_rank_graph_blast(Vector<float>*       p,
                           const Matrix<float> *A,     // column stochastic matrix
                           float alpha, // teleportation constant
                           Descriptor *desc,
                           int _max_iters)
{
    // Get number of vertices
    Index A_nrows = A->nrows();

    // Pagerank vector (p)
    p->fill(1.f/A_nrows);

    // Previous pagerank vector (p_prev)
    Vector<float> p_prev(A_nrows);

    // Temporary pagerank (p_temp)
    Vector<float> p_swap(A_nrows);

    // Residual vector (r)
    Vector<float> r(A_nrows);
    r.fill(1.f);

    Vector<float> const_alpha(A_nrows);
    const_alpha.fill((1.f-alpha)/A_nrows);

    // Temporary residual (r_temp)
    Vector<float> r_temp(A_nrows);

    int iter;
    float error_last = 0.f;
    float error = 1.f;
    Index unvisited = A_nrows;

    for (iter = 0; iter < _max_iters; ++iter)
    {
        unvisited -= static_cast<int>(error);
        error_last = error;
        p_prev = *p;

        // p = A*p + (1-alpha)*1
        vxm<float, float, float, float>(&p_swap, nullptr, second<float>(),
                                        PlusMultipliesSemiring<float>(), &p_prev, A, desc);
        eWiseAdd<float, float, float, float>(p, nullptr, second<float>(),
                                             plus<float>(), &p_swap, &const_alpha, desc);
        // PlusMultipliesSemiring<float>()

        // error = l2loss(p, p_prev)
        eWiseMult<float, float, float, float>(&r, nullptr, second<float>(),
                                              minus<float>(), p, &p_prev, desc);
        //PlusMinusSemiring
        eWiseAdd<float, float, float, float>(&r_temp, nullptr, second<float>(),
                                             multiplies<float>(), &r, &r, desc);
        reduce<float, float>(&error, second<float>(), PlusMonoid<float>(), &r_temp, desc);
        error = sqrt(error);

        float ranks_sum = 0;
        reduce<float, float>(&ranks_sum, second<float>(), PlusMonoid<float>(), p, desc);
        p->print();
        cout << "ranks sum: " << ranks_sum << endl;
    }
}

}
}


// code taken form LAGraph
// add lecense
int LAGraph_VertexCentrality_PageRankGAP (GrB_Vector* centrality, // centrality(i): GAP-style pagerank of node i
                                          // inputs:
                                          LAGraph_Graph<float> *G,        // input graph
                                          int *iters,                     // output: number of iterations taken
                                          int itermax = 100,              // maximum number of iterations (typically 100)
                                          float damping = 0.85,           // damping factor (typically 0.85)
                                          float tol = 1e-4               // stopping tolerance (typically 1e-4) ;
                                          )
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

    const float teleport = (1 - damping) / n;
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

    t->set_name("t");
    r->set_name("r");
    w->set_name("w");
    d->set_name("d");
    //(*centrality)->set_name("centrality");

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GrB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GrB_TRY (GrB_assign (d1, MASK_NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GrB_TRY (GrB_eWiseAdd (d, MASK_NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------
    for ((*iters) = 0 ; (*iters) < itermax; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GrB_TRY (GrB_eWiseMult (w, MASK_NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GrB_TRY (GrB_assign (r, MASK_NULL, NULL, teleport, GrB_ALL, n, NULL)) ;

        // r += A'*w
        SAVE_STATS((GrB_TRY (GrB_mxv (r, MASK_NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32, AT, w, &desc))),
                   "pr_mxv", (sizeof(float)*2 + sizeof(size_t)), 1, AT);

        // t -= r
        GrB_TRY (GrB_assign (t, MASK_NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GrB_TRY (GrB_apply (t, MASK_NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GrB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;

        float ranks_sum = 0;
        GrB_TRY (GrB_reduce (&ranks_sum, NULL, GrB_PLUS_MONOID_FP32, r, NULL));
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

// Алгоритм преобразования LAGraph кода графового алгоритма к коду, совместимому с GB_Kun
// удалить инициализуию LAgraph
// lablas::Vector<Index>* d_out
// *
// изменить маски NULL на MASK_TEMP
// изменить дескриптор
