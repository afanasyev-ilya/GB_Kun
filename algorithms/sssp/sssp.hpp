#pragma once

namespace lablas {

    // diffs from GraphBLAST
    // set_element -> setElement
    // get_nrows -> nrows
    // get_nvals -> nvals
    // null ptr
    // assign<..., index> ...
    // MinimumPlusSemiring<float>() -> lablas::plus<float, float, float>()

/*void sssp(Vector<float>*       v,
          const Matrix<float>* A,
          Index                s,
          Descriptor*          desc,
          int _max_iter)
{
    Index A_nrows;
    A->get_nrows(&A_nrows);

    v->print();

    // Visited vector (use float for now)
    v->fill(std::numeric_limits<float>::max());
    v->set_element(0.f, s);

    // Frontier vectors (use float for now)
    Vector<float> f1(A_nrows);
    Vector<float> f2(A_nrows);

    f1.fill(std::numeric_limits<float>::max());
    f1.set_element(0.f, s);

    // Mask vector
    Vector<float> m(A_nrows);

    Index iter;
    Index f1_nvals = 1;
    float succ = 1.f;

    A->print();


    for (iter = 1; iter <= _max_iter; ++iter)
    {
        cout << "f1: ";
        f1.print();

        vxm<float, float, float, float>(&f2, nullptr, second<Index, float, Index>(),
                                        MinimumPlusSemiring<float>(), &f1, A, desc);

        cout << "v bef: ";
        v->print();
        cout << "f2: ";
        f2.print();

        //eWiseMult<float, float, float, float>(&m, GrB_NULL, GrB_NULL,
        //    PlusLessSemiring<float>(), &f2, v, desc);
        eWiseAdd<float, float, float, float>(&m, nullptr, second<Index, float, Index>(),
                                             lablas::less<float>(), &f2, v, desc);

        cout << "mask: ";
        m.print();

        eWiseAdd<float, float, float, float>(v, nullptr, second<Index, float, Index>(),
                                             lablas::minimum<float>(), v, &f2, desc);

        cout << "v: ";
        v->print();

        // Similar to BFS, except we need to filter out the unproductive vertices
        // here rather than as part of masked vxm
        desc->toggle(GrB_MASK);

        assign<float, float, float>(&f2, &m, second<Index, float, Index>(), std::numeric_limits<float>::max(), GrB_ALL, A_nrows, desc);

        desc->toggle(GrB_MASK);

        f2.swap(&f1);

        Index f1_nvals;
        f1.get_nvals(&f1_nvals);
        reduce<float, float>(&succ, second<Index, float, Index>(), PlusMonoid<float>(), &m, desc);
        cout << "suc: " << succ << endl;

        if (f1_nvals == 0 || succ == 0)
            break;
    }

    v->print();
}*/

void sssp(Vector<float>*       _distances,
          const Matrix<float>* _matrix,
          Index                _source,
          Descriptor*          _desc)
{
    _distances->fill(std::numeric_limits<float>::max());
    _distances->set_element(0.f, _source);

    for (int k = 0; k < _matrix->nrows() - 1; ++k)
    {
        cout << "before: ";
        _distances->print();
        vxm<float, float, float, float>(_distances, NULL, lablas::minimum<float>(),
                MinimumPlusSemiring<float>(), _distances, _matrix, _desc);
        cout << "after: ";
        _distances->print();
    }
}

}