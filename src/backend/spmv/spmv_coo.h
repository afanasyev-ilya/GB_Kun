#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

        template <typename T>
        void SpMV(const MatrixCOO<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y)
        {
            ENT _nz;
            _matrix->get_nz(&_nz);
#pragma omp parallel for schedule(static)
            for(ENT i = 0; i < _nz; i++)
            {
                VNT row = _matrix->get_row()[i];
                VNT col = _matrix->get_col()[i];
                T val = _matrix->get_vals()[i];
        #pragma omp atomic
                _y->get_vals()[row] += val * _x->get_vals()[col];
            }
        }

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
