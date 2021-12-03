#ifndef GB_KUN_BFS_TD_HPP
#define GB_KUN_BFS_TD_HPP
#include "../src/gb_kun.h"

namespace lablas {
namespace algorithm {


void bfs_td(Vector<float>*       v,
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
//        unvisited -= static_cast<int>(succ);

        eWiseMult<float,float,float,float>(v, nullptr, nullptr, FirstWinsSemiring<float>(),
                       &f1, &f1,desc);


        vxm<float, float, float, float>(v, &f1, nullptr, FirstMinSemiring<float>(),
                                        A,&f1 , desc);
        /* Some operations at desc fields */
        //        CHECK(desc->toggle(GrB_MASK));\

        apply<float,float,float>(v,NULL, nullptr,Identity<float>(),&f1,desc);


    }

    cout << "TOP DOWN BFS" << endl;
    v->print();
}

}
}
#endif //GB_KUN_BFS_TD_HPP
