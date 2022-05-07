/*
 * This file uses algorithm implementation from graphblast, which is available under
 * "Apache License 2.0" license. For details, see https://github.com/gunrock/graphblast/blob/master/LICENSE
 * */

#pragma once

namespace lablas {
namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void sssp_bellman_ford_GBTL(Vector<T> *_distances,
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
void sssp_bellman_ford_gbkun(Vector<T> *_distances,
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

}
}
