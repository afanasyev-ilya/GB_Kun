#pragma once

namespace lablas {
namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void sssp_bf_GBTL(Vector<T> *_distances,
                  const Matrix <T> *_matrix,
                  Index _source)
{
    lablas::Descriptor desc;

    _distances->fill(std::numeric_limits<T>::max());
    _distances->set_element(0.f, _source);

    for(Index k = 0; k < _matrix->nrows() - 1; ++k)
    {
        vxm<T, T, T, T>(_distances, NULL, lablas::minimum<T>(), MinimumPlusSemiring<T>(), _distances, _matrix, &desc);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void sssp_bf_gbkun(Vector<T> *_distances,
                   Matrix <T> *_matrix,
                   Index _source)
{
    lablas::Descriptor desc;

    Vector<T> new_distances(_matrix->ncols());
    Vector<T> mask(_matrix->ncols());

    _distances->fill(std::numeric_limits<T>::max());
    _distances->set_element(0.f, _source);
    new_distances.fill(std::numeric_limits<T>::max());
    new_distances.set_element(0.f, _source);

    for(Index k = 0;; k++)
    {
        SAVE_STATS((vxm<T, T, T, T>(&new_distances, NULL, minimum<T>(), MinimumPlusSemiring<T>(),
                _distances, _matrix, &desc));, "sssp_vxm", (sizeof(T)*2 + sizeof(size_t)), 1, _matrix);
        eWiseAdd<T, T, T, T>(&mask, NULL, nullptr, less<T>(), &new_distances, _distances, &desc);

        T succ = 0;
        reduce<T, T>(&succ, second<T>(), PlusMonoid<T>(), &mask, &desc);
        std::cout << "succ " << succ << endl;

        if((succ == 0) || (k >= (_matrix->nrows() - 1)))
        {
            cout << "converged after << " << k << " / " <<  _matrix->nrows() - 1 << " iterations" << endl;
            break;
        }

        _distances->swap(&new_distances);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void sssp_bf_blast(Vector<float> *v,
                   const Matrix <float> *A,
                   Index s,
                   Descriptor *desc)
{
    auto sem_add_op = generic_extract_add(MinimumPlusSemiring<float>());
    auto bin_add_op = generic_extract_add(less<float>());
    std::cout << "sem result: " << sem_add_op(1.0, 2.0) << std::endl;
    std::cout << "bin result: " << bin_add_op(1.0, 2.0) << std::endl;

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
    } else {
        /*std::vector<Index> indices(1, s);
        std::vector<float>  values(1, 0.f);
        f1.build(&indices, &values, 1, GrB_NULL);*/
    }

    // Mask vector
    Vector<float> m(A_nrows);

    Index iter;
    Index f1_nvals = 1;
    float succ = 1.f;

    for (iter = 1; iter <= 1000; ++iter)
    {
        vxm<float, float, float, float>(&f2, nullptr, second<float>(), MinimumPlusSemiring<float>(), &f1, A, desc);

        // CustomLessPlusSemiring<float>()
        eWiseAdd<float, float, float, float>(&m, nullptr, second<float>(), less<float>(), &f2, v, desc);

        //MinimumPlusSemiring<float>()
        eWiseAdd<float, float, float, float>(v, nullptr, second<float>(), minimum<float>(), v, &f2, desc);

        // Similar to BFS, except we need to filter out the unproductive vertices
        // here rather than as part of masked vxm
        desc->toggle(GrB_MASK);
        assign<float, float, float>(&f2, &m, second<float>(),
                                           std::numeric_limits<float>::max(), GrB_ALL, A_nrows, desc);
        desc->toggle(GrB_MASK);

        f2.swap(&f1);

        f1_nvals = f1.nvals();
        reduce<float, float>(&succ, second<float>(), PlusMonoid<float>(), &m, desc);

        if (f1_nvals == 0 || succ == 0)
            break;
    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
