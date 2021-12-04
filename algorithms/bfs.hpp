#ifndef GB_KUN_BFS_HPP
#define GB_KUN_BFS_HPP

#pragma once
#include "../src/gb_kun.h"
namespace lablas {
namespace algorithm {


void bfs(Vector<float>*       v,
         const Matrix<float>* A,
         Index                source,
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

    f1.fill(0.f);
    f1.set_element(1.f, source);
    f1.print();

    float iter;
    float succ = 0.f;
    Index unvisited = A_nrows;
    float max_iter = 10.0;

    A->print();

    for (iter = 1; iter <= max_iter; ++iter) {
        unvisited -= static_cast<int>(succ);

        assign<float, float, float, Index>(v, &f1, nullptr, iter, NULL, A_nrows,
                                           desc);

        vxm<float, float, float, float>(&f2, v, nullptr,
                                        LogicalOrAndSemiring<float>(), A, &f1, desc);

        cout << "after step" << endl;
        f2.print();

        reduce<float, float>(&succ, nullptr, PlusMonoid<float>(), &f1, desc);

        if (succ == 0)
            break;
    }
}

}
}
#endif //GB_KUN_BFS_HPP
