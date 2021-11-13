#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
double SpMV(MatrixLAV<T> &_A, DenseVector<T> &_x, DenseVector<T> &_y)
{
    // load datastructures
    VNT size = _A.get_size();
    ENT *row_ptr = _A.get_row_ptr();
    T *vals = _A.get_vals();
    VNT *col_ids = _A.get_col_ids();

    T *x_vals = _x.get_vals();
    T *y_vals = _y.get_vals();

    // do prefetching
    T *hub_cache;
    VNT *hub_conversion_array = _A.get_hub_conversion_array();
    MemoryAPI::allocate_array(&hub_cache, 64*HUB_VERTICES);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];
        for(int i = 0; i < HUB_VERTICES; i++)
            private_cache[i] = x_vals[hub_conversion_array[i]];
    }

    double t1 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T *private_cache = &hub_cache[tid * HUB_VERTICES];

        #pragma omp for schedule(static)
        for(VNT i = 0; i < size; i++)
        {
            for(ENT j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            {
                VNT col_id = col_ids[j];
                T x_data = 0;
                if(col_id < 0)
                    x_data = private_cache[col_id*(-1)];
                else
                    x_data = x_vals[col_id];
                y_vals[i] += vals[j] * x_data;
            }
        }
    };

    double t2 = omp_get_wtime();

    cout << "SPMV(CSR) perf: " << 2.0*_A.get_nz()/((t2-t1)*1e9) << " GFlop/s" << endl;
    double bw = (3.0*sizeof(VNT)+sizeof(T))*_A.get_nz()/((t2-t1)*1e9);
    cout << "SPMV(CSR) bw: " << bw << " GB/s" << endl;

    MemoryAPI::free_array(hub_cache);
    return bw;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

