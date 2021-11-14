#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(MatrixLAV<T> &_matrix, DenseVector<T> &_x, DenseVector<T> &_y)
{
    // do prefetching
    T *hub_cache;
    VNT *hub_conversion_array = _matrix.hub_conversion_array;
    MemoryAPI::allocate_array(&hub_cache, 64*HUB_VERTICES);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];
        for(int i = 0; i < HUB_VERTICES; i++)
            private_cache[i] = _x.vals[hub_conversion_array[i]];
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];

        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix.size; i++)
        {
            for(ENT j = _matrix.row_ptr[i]; j < _matrix.row_ptr[i + 1]; j++)
            {
                VNT col_id = _matrix.col_ids[j];
                T x_data = 0;
                if(col_id < 0)
                    x_data = private_cache[col_id*(-1)];
                else
                    x_data = _x.vals[col_id];
                _y.vals[i] += _matrix.vals[j] * x_data;
            }
        }
    };

    MemoryAPI::free_array(hub_cache);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

