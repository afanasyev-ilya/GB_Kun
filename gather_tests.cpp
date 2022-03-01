#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/gb_kun.h"
#include <sys/mman.h>
#include <errno.h>
//#include <malloc.h>

#define Index int
#define base_type float

using namespace std;

void saxpy_one_sock(base_type a, base_type * __restrict z, const base_type * __restrict x, const base_type * __restrict y, size_t size)
{
#pragma omp parallel for num_threads(THREADS_PER_SOCKET)
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
    #pragma omp parallel for num_threads(THREADS_PER_SOCKET)
    for(size_t i = 0; i < size; i++)
        result[i] = data[indexes[i]];
}

void gather_copy(const base_type *__restrict data, const Index * __restrict indexes, base_type * __restrict result, size_t size, size_t small_size)
{
    double t1, t2;
    base_type *copy;
    MemoryAPI::allocate_array(&copy, small_size*48);

    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num();

        base_type *loc_data = &copy[tid*small_size];
        for(size_t i = 0; i < small_size; i++)
        {
            loc_data[i] = data[i];
        }
    }

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num();

        base_type *loc_data = &copy[tid*small_size];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
        {
            result[i] = loc_data[indexes[i]];
        }
    };
    t2 = omp_get_wtime();
    cout << "gather copy: " << size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    MemoryAPI::free_array(copy);
}


void gather_copy_12_groups(const base_type *__restrict data, const Index * __restrict indexes, base_type * __restrict result, size_t size, size_t small_size)
{
    double t1, t2;
    base_type *copy;
    MemoryAPI::allocate_array(&copy, small_size*12);

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 4;

        base_type *loc_data = &copy[tid*small_size];
        if((omp_get_thread_num() % 4) == 0)
        {
            for(size_t i = 0; i < small_size; i++)
            {
                loc_data[i] = data[i];
            }
        }
    }
    t2 = omp_get_wtime();
    cout << "prefetch time: " << (t2 - t1)*1000 << " ms" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 4;

        base_type *loc_data = &copy[tid*small_size];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
        {
            result[i] = loc_data[indexes[i]];
        }
    };
    t2 = omp_get_wtime();
    cout << "spmv time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "gather copy 12 groups: " << size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    MemoryAPI::free_array(copy);


    size_t error_count = 0;
    #pragma omp parallel for num_threads(THREADS_PER_SOCKET) reduction(+: error_count)
    for(size_t i = 0; i < size; i++)
    {
        if(result[i] != data[indexes[i]])
            error_count++;
    }
    cout << "12 group check: " << error_count << " / " << size << endl;
}


void scatter_copy(base_type *data, const Index * __restrict indexes, base_type * __restrict result, size_t size, size_t small_size)
{
    double t1, t2;
    base_type *copy;
    MemoryAPI::allocate_array(&copy, small_size*48);

    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num();
        base_type *loc_data = &copy[tid*small_size];
        for(size_t i = 0; i < small_size; i++)
        {
            loc_data[i] = 0;
        }
    }

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num();

        base_type *loc_data = &copy[tid*small_size];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
        {
            loc_data[indexes[i]] = result[i];
        }
    };
    t2 = omp_get_wtime();
    cout << "scatter copy: " << size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    for(int tid = 0; tid < 48; tid++)
    {
        base_type *loc_data = &copy[tid*small_size];

        #pragma omp parallel for
        for(size_t i = 0; i < small_size; i++)
        {
            data[i] = max(loc_data[i], data[i]);
        }
    }

    MemoryAPI::free_array(copy);
}

void scatter_copy_12_groups(base_type *data, const Index * __restrict indexes, base_type * __restrict result, size_t size, size_t small_size)
{
    double t1, t2;
    base_type *copy;
    MemoryAPI::allocate_array(&copy, small_size*12);

    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 4;
        base_type *loc_data = &copy[tid*small_size];
        for(size_t i = 0; i < small_size; i++)
        {
            loc_data[i] = 0;
        }
    }

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 4;

        base_type *loc_data = &copy[tid*small_size];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
        {
            loc_data[indexes[i]] = result[i];
        }
    };
    t2 = omp_get_wtime();
    cout << "scatter copy 12 groups: " << size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    for(int tid = 0; tid < 12; tid++)
    {
        base_type *loc_data = &copy[tid*small_size];

        #pragma omp parallel for
        for(size_t i = 0; i < small_size; i++)
        {
            data[i] = max(loc_data[i], data[i]);
        }
    }

    MemoryAPI::free_array(copy);
}

void scatter_copy_6_groups(base_type *data, const Index * __restrict indexes, base_type * __restrict result, size_t size, size_t small_size)
{
    double t1, t2;
    base_type *copy;
    MemoryAPI::allocate_array(&copy, small_size*6);

    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 8;
        base_type *loc_data = &copy[tid*small_size];
        for(size_t i = 0; i < small_size; i++)
        {
            loc_data[i] = 0;
        }
    }

    t1 = omp_get_wtime();
    #pragma omp parallel num_threads(THREADS_PER_SOCKET)
    {
        int tid = omp_get_thread_num() / 8;

        base_type *loc_data = &copy[tid*small_size];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
        {
            loc_data[indexes[i]] = result[i];
        }
    };
    t2 = omp_get_wtime();
    cout << "scatter copy 6 groups: " << size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

    for(int tid = 0; tid < 6; tid++)
    {
        base_type *loc_data = &copy[tid*small_size];

        #pragma omp parallel for
        for(size_t i = 0; i < small_size; i++)
        {
            data[i] = max(loc_data[i], data[i]);
        }
    }

    MemoryAPI::free_array(copy);
}

void scatter_one_sock(base_type *data, const Index * __restrict indexes, base_type * __restrict result, size_t size)
{
    #pragma omp parallel for num_threads(THREADS_PER_SOCKET)
    for(size_t i = 0; i < size; i++)
        data[indexes[i]] = result[i];
}

void reinit_data(base_type *data, size_t current_radius)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < current_radius; i++)
            data[i] = rand_r(&myseed) % 1024;
    }
}


#include <chrono>
#include <thread>

Index main(void)
{
    cout << "threads: " << omp_get_max_threads() << endl;

    size_t size = 1610612736; // 6.4 GB each
    /*base_type *x, *y, *z;
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

    for(Index i = 0; i < 10; i++) {
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

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));*/

    base_type *result, *data;
    Index *indexes;

    size_t large_size = size;

    const int inner_runs = 80;
    const int num_tests = 9;//11;
    size_t rads[num_tests] = {512*1024/sizeof(base_type),
                              1024*1024/sizeof(base_type),
                              1*1024*1024/sizeof(base_type),
                              2*1024*1024/sizeof(base_type),
                              4*1024*1024/sizeof(base_type),
                              8*1024*1024/sizeof(base_type),
                              16*1024*1024/sizeof(base_type),
                              32*1024*1024/sizeof(base_type),
                              64*1024*1024/sizeof(base_type)
                              /*128*1024*1024/sizeof(base_type),
                              256*1024*1024/sizeof(base_type)*/};

    cout << "num_tests: " << num_tests << endl;
    cout << "large size is " << large_size * sizeof(Index) / (1024*1024) << " MB" << endl;
    for(int idx = 0; idx < num_tests; idx++)
    {
        cout << "num_tests: " << idx << " " << num_tests << endl;
        size_t current_radius = rads[idx];
        cout << "Rad size is " << current_radius * sizeof(Index) / (1024) << " KB" << endl;
        cout << "Rad size is " << ((double)current_radius * sizeof(Index)) / (1024*1024) << " MB" << endl;

        MemoryAPI::allocate_array(&result, large_size);
        MemoryAPI::allocate_array(&indexes, large_size);
        //MemoryAPI::allocate_array(&data, current_radius);
        //data = memalign(2*1024*1024, current_radius*sizeof(base_type));
        data = (base_type*)aligned_alloc(2*1024*1024, current_radius*sizeof(base_type));
        /*if(madvise(data, current_radius, MADV_HUGEPAGE) == -1)
        {
            if (errno == EACCES)
                cout << " EACCES " << endl;
            if (errno == EAGAIN)
                cout << " EAGAIN " << endl;
            if (errno == EBADF)
                cout << " EBADF " << endl;
            if (errno == EINVAL)
                cout << " EINVAL " << endl;
            if (errno == EINVAL)
                cout << " EINVAL " << endl;
            if (errno == EIO)
                cout << " EIO " << endl;
            if (errno == ENOMEM)
                cout << " ENOMEM " << endl;
            if (errno == EAGAIN)
                cout << " EAGAIN " << endl;
            if (errno == EPERM)
                cout << " EPERM " << endl;
        }
        if(madvise(data, current_radius,  MADV_HUGEPAGE | MADV_RANDOM) == -1)
        {
            if (errno == EACCES)
                cout << " EACCES " << endl;
            if (errno == EAGAIN)
                cout << " EAGAIN " << endl;
            if (errno == EBADF)
                cout << " EBADF " << endl;
            if (errno == EINVAL)
                cout << " EINVAL " << endl;
            if (errno == EINVAL)
                cout << " EINVAL " << endl;
            if (errno == EIO)
                cout << " EIO " << endl;
            if (errno == ENOMEM)
                cout << " ENOMEM " << endl;
            if (errno == EAGAIN)
                cout << " EAGAIN " << endl;
            if (errno == EPERM)
                cout << " EPERM " << endl;
        }*/

        #pragma omp parallel
        {
            unsigned int myseed = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < large_size; i++)
            {
                indexes[i] = (Index) rand_r(&myseed) % current_radius;
                result[i] = i % 1024;
            }
        }

        reinit_data(data, current_radius);

        double t1, t2;

        for(int i = 0; i < inner_runs; i++)
        {
            t1 = omp_get_wtime();
            gather_one_sock(data, indexes, result, large_size);
            t2 = omp_get_wtime();
            cout << "gather one sock: " << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;
        }

        /*t1 = omp_get_wtime();
        gather_one_sock(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << "gather one sock: " << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

        gather_copy(data, indexes, result, large_size, current_radius);

        gather_copy_12_groups(data, indexes, result, large_size, current_radius);

        reinit_data(data, current_radius);

        t1 = omp_get_wtime();
        scatter_one_sock(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << "scatter one sock: " << current_radius * sizeof(Index) / (1024) << "KB " << large_size * (sizeof(Index) + 2*sizeof(base_type)) / ((t2 - t1)*1e9) << " GB/s" << endl;

        reinit_data(data, current_radius);
        scatter_copy(data, indexes, result, large_size, current_radius);

        reinit_data(data, current_radius);
        scatter_copy_12_groups(data, indexes, result, large_size, current_radius);

        reinit_data(data, current_radius);
        scatter_copy_6_groups(data, indexes, result, large_size, current_radius);*/

        MemoryAPI::free_array(indexes);
        MemoryAPI::free_array(result);
        MemoryAPI::free_array(data);
        cout << endl << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }

    return 0;
}