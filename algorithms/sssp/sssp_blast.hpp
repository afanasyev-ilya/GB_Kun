/**
  @file sssp_blast.hpp
  @author S.krymskiy
  @version Revision 1.1
  @brief SSSP blast algorithm.
  @details Detailed description.
  @date May 12, 2022
*/

#pragma once

//! Lablas namespace

namespace lablas {

    //! Algorithm namespace

    namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * sssp_bellman_ford_blast function.
 * @brief The function does...
 * @param v v
 * @param A A
 * @param s s
 * @param desc desc
*/

void sssp_bellman_ford_blast(Vector<float> *v,
                             const Matrix<float> *A,
                             Index s,
                             Descriptor *desc)
{
    const auto previous_omp_dynamic = omp_get_dynamic();
    int previous_omp_threads;
    #pragma omp parallel
    {
        #pragma omp single
        previous_omp_threads = omp_get_num_threads();
    }
    if (previous_omp_threads == 96) {
        omp_set_num_threads(48);
    }

    if (previous_omp_threads == 128) {
        omp_set_num_threads(64);
    }

    Index A_nrows = A->nrows();

    // Visited vector (use float for now)
    v->fill(std::numeric_limits<float>::max());
    v->set_element(0.f, s);

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);

    Desc_value desc_value;

    // Visited vector (use float for now)
    if (true)
    {
        f1.fill(std::numeric_limits<float>::max());
        f1.set_element(0.f, s);
    }
    else
    {
        /*std::vector<Index> indices(1, s); // TODO
        std::vector<float>  values(1, 0.f);
        f1.build(&indices, &values, 1, GrB_NULL);*/
    }

    // Mask vector
    Vector<float> m(A_nrows);

    Index iter = 0;
    Index f1_nvals = 1;
    float succ = 1.f;
    Index max_iters = A_nrows;

    for (iter = 1; iter <= max_iters; ++iter)
    {
        vxm<float, float, float, float>(&f2, nullptr, GrB_NULL, MinimumPlusSemiring<float>(), &f1, A, desc);

        eWiseAdd<float, float, float, float>(&m, nullptr, GrB_NULL,
                                             CustomLessPlusSemiring<float>(), &f2, v, desc);

        eWiseAdd<float, float, float, float>(v, nullptr, GrB_NULL,
                                             MinimumPlusSemiring<float>(), v, &f2, desc);

        // Similar to BFS, except we need to filter out the unproductive vertices
        // here rather than as part of masked vxm
        desc->toggle(GrB_MASK);
        assign<float, float, float>(&f2, &m, GrB_NULL, std::numeric_limits<float>::max(),
                                    GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);

        f2.swap(&f1);

        f1_nvals = f1.nvals();
        reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &m, desc);

        if (f1_nvals == 0 || succ == 0)
            break;
    }
    std::cout << "sssp did " << iter << " iterations" << std::endl;

    omp_set_dynamic(previous_omp_dynamic);
    omp_set_num_threads(previous_omp_threads);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}