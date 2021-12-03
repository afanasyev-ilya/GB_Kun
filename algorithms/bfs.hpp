#ifndef GB_KUN_BFS_HPP
#define GB_KUN_BFS_HPP

#pragma once
#include "../src/gb_kun.h"
namespace lablas {
namespace algorithm {


void bfs(Vector<float>*       v,
         const Matrix<float>* A,
         Index                s,
         Descriptor*          desc)
{

    VNT A_nrows;
    A->get_nrows(&A_nrows);

    // Visited vector (use float for now)
    v->fill(0.f);

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);
    const Vector<float> f3(A_nrows);

    Desc_value desc_value;
    desc->get(GrB_MXVMODE, &desc_value);
    if (desc_value == GrB_PULLONLY) {
        f1.fill(0.f);
        f1.set_element(1.f, s);
    } else {
        std::vector<Index> indices(1, s);
        std::vector<float>  values(1, 1.f);
        /* Creating SPARSE VECTOR!!!! - Dense can be created without indices param */
        f1.build(&indices, &values, 1);
    }

    float iter;
    float succ = 0.f;
    Index unvisited = A_nrows;
    float max_iter = 10.0;

    for (iter = 1; iter <= max_iter; ++iter) {
        //        if (iter > 1) {
        //            std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        //                    "push" : "pull";
        //        }
        unvisited -= static_cast<int>(succ);

        assign<float, float, float, Index>(v, &f1, nullptr, iter, NULL, A_nrows,
                                           desc);
        /* Some operations at desc fields */
        //        CHECK(desc->toggle(GrB_MASK));\

        mxv<float, float, float, float>(&f2, v, nullptr,
                                        LogicalOrAndSemiring<float>(), A, &f1, desc);

        PlusMultipliesSemiring<float, float, float> a;
        a.identity();

        //        CHECK(desc->toggle(GrB_MASK));
        //
        //        CHECK(f2.swap(&f1));
        reduce<float, float>(&succ, nullptr, PlusMonoid<float>(), &f1, desc);

    }

    v->print();

    //    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
    //            "push" : "pull";
    //         }

}

}
}
#endif //GB_KUN_BFS_HPP
