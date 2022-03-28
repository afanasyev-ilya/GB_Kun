#pragma once
#include "../../src/gb_kun.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GrB_Matrix lablas::Matrix<int>*
#define GrB_Vector lablas::Vector<int>*
#define MASK_NULL static_cast<const lablas::Vector<int>*>(NULL)

int LG_BreadthFirstSearch_vanilla(GrB_Vector *level,
                                  GrB_Vector *parent,
                                  LAGraph_Graph<int> *G,
                                  GrB_Index src,
                                  bool pushpull)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    lablas::Vector<bool> *frontier = NULL;     // the current frontier
    GrB_Vector l_parent = NULL;     // parent vector
    GrB_Vector l_level = NULL;      // level vector
    lablas::Descriptor desc;

    //--------------------------------------------------------------------------
    // get the problem size and properties
    //--------------------------------------------------------------------------
    GrB_Matrix A = G->A;

    GrB_Index n;
    GrB_TRY( GrB_Matrix_nrows (&n, A) );
    assert(src < n && "invalid source node");

    // only the level is needed

    // create a sparse boolean vector frontier, and set frontier(src) = true
    GrB_TRY (GrB_Vector_new(&frontier, GrB_BOOL, n)) ;
    GrB_TRY (GrB_Vector_setElement(frontier, true, src)) ;
    frontier->print();

    // create the level vector. v(i) is the level of node i
    // v (src) = 0 denotes the source node
    GrB_TRY (GrB_Vector_new(&l_level, GrB_INT32, n)) ;

    //--------------------------------------------------------------------------
    // BFS traversal and label the nodes
    //--------------------------------------------------------------------------
    GrB_Index nq = 1; // number of nodes in the current level
    GrB_Index last_nq = 0;
    GrB_Index current_level = 1;
    GrB_Index nvals = 1;

    // {!mask} is the set of unvisited nodes
    GrB_Vector mask = l_level ;

    // parent BFS
    do
    {
        // assign levels: l_level<s(frontier)> = current_level
        GrB_TRY( GrB_assign(l_level, frontier, NULL, current_level, GrB_ALL, n, GrB_DESC_S) );
        l_level->print();
        ++current_level;

        // frontier = kth level of the BFS
        // mask is l_parent if computing parent, l_level if computing just level
        GrB_TRY( GrB_vxm(frontier, mask, NULL, lablas::LogicalOrAndSemiring<bool>(), frontier, A, GrB_DESC_RSC) );
        frontier->print();

        // done if frontier is empty
        GrB_TRY( GrB_Vector_nvals(&nvals, frontier) );
    } while (nvals > 0);

    //l_level->force_to_dense();

    (*level ) = l_level;
    return (0);
}

#undef GrB_Matrix
#undef GrB_Vector
#undef MASK_NULL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace algorithm {

void bfs_blast(Vector<float>*       v,
               const Matrix<float> *A,
               Index s,
               Descriptor *desc)
{
    Index A_nrows = A->nrows();

    // Visited vector (use float for now)
    v->fill(0.f);

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);

    Desc_value desc_value;
    desc->get(GrB_MXVMODE, &desc_value);
    if (true/*desc_value == GrB_PULLONLY*/)
    {
        f1.fill(0.f);
        f1.set_element(1.f, s);
    }
    else
    {
        /*std::vector<Index> indices(1, s);
        std::vector<float>  values(1, 1.f);
        CHECK(f1.build(&indices, &values, 1, GrB_NULL));*/
    }

    Index iter = 0;
    float succ = 0.f;
    Index unvisited = A_nrows;
    float gpu_tight_time = 0.f;
    Index max_iters = A_nrows;

    for (iter = 1; iter <= max_iters; ++iter) {
        unvisited -= static_cast<int>(succ);
        assign<float, float, float>(v, &f1, second<float>()/*GrB_NULL*/, iter, GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);
        vxm<float, float, float, float>(&f2, v, second<float>()/*GrB_NULL*/,LogicalOrAndSemiring<float>(), &f1, A, desc);
        desc->toggle(GrB_MASK);

        f2.swap(&f1);
        reduce<float, float>(&succ, second<float>()/*GrB_NULL*/, PlusMonoid<float>(), &f1, desc);

        if (succ == 0)
            break;
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

