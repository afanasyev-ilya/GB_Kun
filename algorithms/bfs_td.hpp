#ifndef GB_KUN_BFS_TD_HPP
#define GB_KUN_BFS_TD_HPP
#include "../src/gb_kun.h"

namespace lablas {
namespace algorithm {


void bfs_td(Vector<float>*    levels,
         const Matrix<float>* A,
         Index                source,
         Descriptor*          desc)
{
    /*Index const N(A->nrows());

    Vector<bool> wavefront(N);
    wavefront.set_element(source, true);

    Index depth(0);
    while (depth < 10)
    {
        // Increment the level
        ++depth;

        // Apply the level to all newly visited nodes
        apply(levels, nullptr, Plus<Index>(),
                //[depth](auto arg) { return arg * depth; },
                   std::bind(grb::Times<grb::IndexType>(),
                             depth,
                             std::placeholders::_1),
                   wavefront);

        grb::mxv(wavefront, complement(levels),
                 grb::NoAccumulate(),
                 grb::LogicalSemiring<bool>(),
                 transpose(graph), wavefront, grb::REPLACE);
    }

    Index const N(A->get_nrows());

    // assert parent_list is N-vector
    // assert source is in proper range
    // assert parent_list ScalarType is grb::IndexType

    // create index ramp for index_of() functionality
    Vector<Index> index_ramp(N);
    for (Index i = 0; i < N; ++i)
    {
        index_ramp.setElement(i, i);
    }

    // initialize wavefront to source node.
    Vector<Index> wavefront(N);
    wavefront.setElement(source, 1UL);

    // set root parent to self;
    parent_list.clear();
    parent_list.setElement(source, source);

    while (wavefront.nvals() > 0)
    {
        eWiseMult<float,float,float,float>(v, nullptr, nullptr, FirstWinsSemiring<float>(),
                       &f1, &f1,desc);


        vxm<float, float, float, float>(v, &f1, nullptr, FirstMinSemiring<float>(),
                                        A,&f1 , desc);

        apply<float,float,float>(v,NULL, nullptr,Identity<float>(),&f1,desc);
    }

    cout << "TOP DOWN BFS" << endl;
    v->print();*/
}

}
}
#endif //GB_KUN_BFS_TD_HPP
