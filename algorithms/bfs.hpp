#pragma once

void bfs(Vector<float>*       v,
         const Matrix<float>* A,
         Index                s,
         Descriptor*          desc)
{
    Index A_nrows;
    CHECK(A->nrows(&A_nrows));

    // Visited vector (use float for now)
    CHECK(v->fill(0.f));

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);

    Desc_value desc_value;
    CHECK(desc->get(GrB_MXVMODE, &desc_value));
    if (desc_value == GrB_PULLONLY) {
        CHECK(f1.fill(0.f));
        CHECK(f1.setElement(1.f, s));
    } else {
        std::vector<Index> indices(1, s);
        std::vector<float>  values(1, 1.f);
        CHECK(f1.build(&indices, &values, 1, GrB_NULL));
    }

    float iter;
    float succ = 0.f;
    Index unvisited = A_nrows;

    for (iter = 1; iter <= desc->descriptor_.max_niter_; ++iter) {
        if (iter > 1) {
            std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
                                   "push" : "pull";
        }
        unvisited -= static_cast<int>(succ);
        gpu_tight.Start();

        assign<float, float, float, Index>(v, &f1, GrB_NULL, iter, GrB_ALL, A_nrows,
                                           desc);
        CHECK(desc->toggle(GrB_MASK));
        vxm<float, float, float, float>(&f2, v, GrB_NULL,
                                        LogicalOrAndSemiring<float>(), &f1, A, desc);
        CHECK(desc->toggle(GrB_MASK));

        CHECK(f2.swap(&f1));
        reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &f1, desc);

        if (desc->descriptor_.debug())
            std::cout << "succ: " << succ << std::endl;
        if (succ == 0)
            break;
    }
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
                           "push" : "pull";
}