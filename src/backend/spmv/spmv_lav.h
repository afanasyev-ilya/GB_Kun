#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend {

template <typename T, typename SemiringT>
void SpMV(const MatrixLAV<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    // do prefetching
    T *hub_cache;
    VNT *hub_conversion_array = _matrix->hub_conversion_array;
    MemoryAPI::allocate_array(&hub_cache, 64*HUB_VERTICES);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];
        for(int i = 0; i < HUB_VERTICES; i++)
            private_cache[i] = x_vals[hub_conversion_array[i]];
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];

        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix->size; i++)
        {
            for(ENT j = _matrix->row_ptr[i]; j < _matrix->row_ptr[i + 1]; j++)
            {
                VNT col_id = _matrix->col_ids[j];
                T x_data = 0;
                if(col_id < 0)
                    x_data = private_cache[col_id*(-1)];
                else
                    x_data = x_vals[col_id];
                y_vals[i] += _matrix->vals[j] * x_data;
            }
        }
    };

    MemoryAPI::free_array(hub_cache);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

