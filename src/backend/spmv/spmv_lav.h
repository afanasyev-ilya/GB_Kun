#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend {

template <typename T>
void SpMV(const MatrixLAV<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y)
{
    // do prefetching
    T *hub_cache;
    VNT matrix_size;
    _matrix->get_size(&matrix_size);
    const VNT *hub_conversion_array = _matrix->get_hub();
    const T* vals = _x->get_vals();
    MemoryAPI::allocate_array(&hub_cache, 64*HUB_VERTICES);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];
        for(int i = 0; i < HUB_VERTICES; i++)
            private_cache[i] = vals[hub_conversion_array[i]];
    }

#pragma omp parallel
{
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];

    #pragma omp for schedule(static)
        for(VNT i = 0; i < matrix_size; i++)
        {
            for(ENT j = _matrix->get_row()[i]; j < _matrix->get_row()[i + 1]; j++)
            {
                VNT col_id = _matrix->get_col()[j];
                T x_data = 0;
                if(col_id < 0)
                    x_data = private_cache[col_id*(-1)];
                else
                    x_data = _x->get_vals()[col_id];
                _y->get_vals()[i] += _matrix->get_vals()[j] * x_data;
            }
        }
    };

    MemoryAPI::free_array(hub_cache);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

