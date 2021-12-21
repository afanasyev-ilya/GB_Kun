#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/helpers/memory_API/memory_API.h"

#define Index int
#define base_type float

using namespace std;

void saxpy_one_sock(base_type a, base_type * __restrict z, const base_type * __restrict x, const base_type * __restrict y, size_t size)
{
    #pragma omp parallel for num_threads(48)
    for(size_t i = 0; i < size; i++)
        z[i] = a*x[i] + y[i];
}

void saxpy_both_sock(base_type a, base_type * __restrict z, const base_type * __restrict x, const base_type * __restrict y, size_t size)
{
    #pragma omp parallel for num_threads(96)
    for(size_t i = 0; i < size; i++)
        z[i] = a*x[i] + y[i];
}

void gather_one_sock(const base_type *__restrict data, const Index * __restrict indexes, base_type * __restrict result, size_t size)
{
    #pragma omp parallel for num_threads(48)
    for(size_t i = 0; i < size; i++)
        result[i] = data[indexes[i]];
}

void gather_two_sock(const base_type *__restrict data, const Index * __restrict indexes, base_type * __restrict result, size_t size)
{
    #pragma omp parallel for num_threads(96)
    for(size_t i = 0; i < size; i++)
        result[i] = data[indexes[i]];
}

void gather_numa(base_type *__restrict data, const Index * __restrict indexes, base_type * __restrict result, size_t size,
                 size_t small_size)
{
    #pragma omp parallel num_threads(48)
    {
        unsigned int myseed = omp_get_thread_num();
        Index sock = omp_get_thread_num() / 48;
        const base_type *__restrict local_data = &(data[small_size * 0]);
        base_type *__restrict remote_data = &(data[small_size * 1]);

        #pragma omp for
        for (size_t i = 0; i < small_size; i++)
            remote_data[i] = local_data[i];
    }

    #pragma omp parallel num_threads(96)
    {
        Index sock = omp_get_thread_num() / 48;
        const base_type *__restrict local_data = &(data[small_size * sock]);

        #pragma omp for
        for(size_t i = 0; i < size; i++)
            result[i] = local_data[indexes[i]];
    }

}

Index main(void)
{
    cout << "threads: " << omp_get_max_threads() << endl;

    size_t size = 1610612736; // 6.4 GB each
    base_type *x, *y, *z;
    MemoryAPI::allocate_array(&x, size);
    MemoryAPI::allocate_array(&y, size);
    MemoryAPI::allocate_array(&z, size);

    #pragma omp parallel for
    for (Index i = 0; i < size; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 3.0f;
    }

    for(Index i = 0; i < 4; i++) {
        double t1 = omp_get_wtime();
        saxpy_one_sock(2.0f, z, x, y, size);
        double t2 = omp_get_wtime();
        cout << size * sizeof(base_type) * 3.0 / ((t2 - t1)*1e9) << " GB/s (one sock)" << endl;

        t1 = omp_get_wtime();
        saxpy_both_sock(2.0f, z, x, y, size);
        t2 = omp_get_wtime();
        cout << size * sizeof(base_type) * 3.0 / ((t2 - t1)*1e9) << " GB/s (both sock)" << endl;
    }
    free(x);
    free(y);
    free(z);

    base_type *result, *data;
    Index *indexes;
    
    size_t large_size = size;
    
    const int num_tests = 11;
    size_t rads[num_tests] = {128*1024/sizeof(base_type),
                              256*1024/sizeof(base_type),
                    1024*1024/sizeof(base_type),
                   2*1024*1024/sizeof(base_type),
                   4*1024*1024/sizeof(base_type),
                   8*1024*1024/sizeof(base_type),
                   16*1024*1024/sizeof(base_type),
                   32*1024*1024/sizeof(base_type),
                   64*1024*1024/sizeof(base_type),
                   128*1024*1024/sizeof(base_type),
                   256*1024*1024/sizeof(base_type)};

    cout << "num_tests: " << num_tests << endl;
    cout << "large size is " << large_size * sizeof(Index) / (1024*1024) << " MB" << endl;
    for(int idx = 0; idx < num_tests; idx++)
    {
        cout << "num_tests: " << idx << " " << num_tests << endl;
        size_t current_radius = rads[idx];
        cout << "Rad size is " << current_radius * sizeof(Index) / (1024) << " KB" << endl;
        
        MemoryAPI::allocate_array(&result, large_size);
        MemoryAPI::allocate_array(&indexes, large_size);
        MemoryAPI::allocate_array(&data, current_radius*2);

        #pragma omp parallel
        {
            unsigned int myseed = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < large_size; i++)
            {
                indexes[i] = (Index) rand_r(&myseed) % current_radius;
                result[i] = 0;
            }
        }

        #pragma omp parallel
        {
            unsigned int myseed = omp_get_thread_num();
            Index sock = omp_get_thread_num() / 48;
            base_type *__restrict local_data = &(data[current_radius * sock]);

            #pragma omp for
            for (size_t i = 0; i < current_radius; i++)
                local_data[i] = rand_r(&myseed);
        }

        double t1 = omp_get_wtime();
        gather_one_sock(data, indexes, result, large_size);
        double t2 = omp_get_wtime();
        cout << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s (one sock)" << endl;

        t1 = omp_get_wtime();
        gather_two_sock(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s (both sock)" << endl;

        t1 = omp_get_wtime();
        gather_numa(data, indexes, result, large_size, current_radius);
        t2 = omp_get_wtime();
        cout << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s (numa)" << endl;

        MemoryAPI::free_array(indexes);
        MemoryAPI::free_array(result);
        MemoryAPI::free_array(data);
        cout << endl;
    }

    return 0;
}
