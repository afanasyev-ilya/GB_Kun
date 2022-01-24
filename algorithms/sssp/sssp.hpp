#pragma once

namespace lablas {

    // diffs from GraphBLAST
    // set_element -> setElement
    // get_nrows -> nrows
    // get_nvals -> nvals
    // null ptr

void sssp(Vector<float>*       v,
          const Matrix<float>* A,
          Index                s,
          Descriptor*          desc,
          int _max_iter)
{
    Index A_nrows;
    A->get_nrows(&A_nrows);

    // Visited vector (use float for now)
    v->fill(std::numeric_limits<float>::max());
    v->set_element(0.f, s);

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);

    Desc_value desc_value;
    desc->get(GrB_MXVMODE, &desc_value);

    // Visited vector (use float for now)
    if (desc_value == GrB_PULLONLY)
    {
        f1.fill(std::numeric_limits<float>::max());
        f1.set_element(0.f, s);
    }
    else
    {
        std::vector<Index> indices(1, s);
        std::vector<float>  values(1, 0.f);
        // f1.build(&indices, &values, 1, GrB_NULL);
        f1.build(&indices, &values, 1);
    }

    // Mask vector
    Vector<float> m(A_nrows);

    Index iter;
    Index f1_nvals = 1;
    float succ = 1.f;

    for (iter = 1; iter <= _max_iter; ++iter)
    {
        vxm<float, float, float, float>(&f2, nullptr, GrB_NULL,
                                        MinimumPlusSemiring<float>(), &f1, A, desc);

        //eWiseMult<float, float, float, float>(&m, GrB_NULL, GrB_NULL,
        //    PlusLessSemiring<float>(), &f2, v, desc);
        eWiseAdd<float, float, float, float>(&m, nullptr, GrB_NULL,
                                             CustomLessPlusSemiring<float>(), &f2, v, desc);

        eWiseAdd<float, float, float, float>(v, nullptr, GrB_NULL,
                                             MinimumPlusSemiring<float>(), v, &f2, desc);

        // Similar to BFS, except we need to filter out the unproductive vertices
        // here rather than as part of masked vxm
        desc->toggle(GrB_MASK);

        assign<float, float, float, Index>(&f2, &m, second<float, float, float>(), std::numeric_limits<float>::max(), GrB_ALL, A_nrows, desc);

        desc->toggle(GrB_MASK);

        f2.swap(&f1);

        Index f1_nvals;
        f1.get_nvals(&f1_nvals);
        reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &m, desc);

        if (f1_nvals == 0 || succ == 0)
            break;
    }
}

}