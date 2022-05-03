#pragma once

namespace lablas {
namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void sssp_bellman_ford_blast(Vector<float> *v,
                             const Matrix<float> *A,
                             Index s,
                             Descriptor *desc)
{
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
        std::cout << "1" << std::endl;
        vxm<float, float, float, float>(&f2, nullptr, GrB_NULL, MinimumPlusSemiring<float>(), &f1, A, desc);
        std::cout << "2" << std::endl;

        eWiseAdd<float, float, float, float>(&m, nullptr, GrB_NULL,
                                             CustomLessPlusSemiring<float>(), &f2, v, desc);

        eWiseAdd<float, float, float, float>(v, nullptr, GrB_NULL,
                                             MinimumPlusSemiring<float>(), v, &f2, desc);

        // Similar to BFS, except we need to filter out the unproductive vertices
        // here rather than as part of masked vxm
        desc->toggle(GrB_MASK);
        desc->toggle(GrB_OUTPUT);
        assign<float, float, float>(&f2, &m, GrB_NULL, std::numeric_limits<float>::max(),
                                    GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);
        desc->toggle(GrB_OUTPUT);
        std::cout << "it done" << std::endl;

        f2.swap(&f1);

        f1_nvals = f1.nvals();
        reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &m, desc);

        if (f1_nvals == 0 || succ == 0)
            break;
    }
    std::cout << "sssp did " << iter << " iterations" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}