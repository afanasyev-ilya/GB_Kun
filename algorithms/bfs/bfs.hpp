#pragma once
#include "../../src/gb_kun.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GrB_Matrix lablas::Matrix<int>*
#define GrB_Vector lablas::Vector<int>*
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

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
    GrB_Index nq = 1 ;          // number of nodes in the current level
    GrB_Index last_nq = 0 ;
    GrB_Index current_level = 1;
    GrB_Index nvals = 1;

    // {!mask} is the set of unvisited nodes
    GrB_Vector mask = l_level ;

    // parent BFS
    do
    {
        cout << " ------------------- " << endl;
        // assign levels: l_level<s(frontier)> = current_level
        GrB_TRY( GrB_assign(l_level, frontier, NULL, current_level, GrB_ALL, n, &desc) );
        l_level->print();
        ++current_level;

        // frontier = kth level of the BFS
        // mask is l_parent if computing parent, l_level if computing just level
        cout << "mask ";
        mask->print();
        desc.set(GrB_MASK, GrB_DEFAULT);
        GrB_TRY( GrB_vxm(frontier, mask, NULL, LAGraph_structural_bool, frontier, A, &desc) );
        cout << "frontier after vxm ";
        frontier->print();

        // done if frontier is empty
        GrB_TRY( GrB_Vector_nvals(&nvals, frontier) );
    } while (nvals > 0);


    (*level ) = l_level;
    return (0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int my_BreadthFirstSearch_vanilla(GrB_Vector *levels,
                                  LAGraph_Graph<int> *G,
                                  GrB_Index src)
{
    lablas::Descriptor desc;
    GrB_Matrix A = G->A;
    GrB_Index n;
    GrB_TRY( GrB_Matrix_nrows (&n, A) );

    GrB_Vector f1 = NULL, *f2 = NULL, *v = NULL;
    GrB_TRY(GrB_Vector_new(&f1, GrB_INT32, n));
    GrB_TRY(GrB_Vector_new(&f2, GrB_INT32, n));
    GrB_TRY(GrB_Vector_new(&v, GrB_INT32, n));

    GrB_TRY(GrB_assign(f1, MASK_NULL, NULL, 0, GrB_ALL, n, NULL));
    GrB_TRY(GrB_assign(f2, MASK_NULL, NULL, 0, GrB_ALL, n, NULL));
    GrB_TRY(GrB_assign(v, MASK_NULL, NULL, 0, GrB_ALL, n, NULL));
    GrB_TRY (GrB_Vector_setElement(f1, 1, src)) ;

    int iter = 0;
    int succ = 0;
    do {
        GrB_TRY(GrB_assign(v, f1, NULL, iter, GrB_ALL, n, NULL));
        GrB_TRY( GrB_vxm(f2, v, NULL, lablas::LogicalOrAndSemiring<int>(), f1, A, &desc));

        GrB_Vector tmp = f2;
        f2 = f1;
        f1 = tmp;

        succ = 0;
        f1->print();
        GrB_TRY (GrB_reduce (&succ, NULL, GrB_PLUS_MONOID_INT32, f1, NULL)) ;
        cout << "succ: " << succ << "/" << n << endl;

        iter++;
    } while(succ > 0);

    v->print();
    *levels = v;
}

#undef GrB_Matrix
#undef GrB_Vector
#undef MASK_NULL

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

