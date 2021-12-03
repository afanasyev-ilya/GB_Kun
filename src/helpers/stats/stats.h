#pragma once

#include <stdio.h>
#include <sched.h>
#include <omp.h>

void print_omp_stats()
{
    /*#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int cpu_num = sched_getcpu();
        printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
    }

    int max_thread = 0, max_core = 0;
    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int cpu_num = sched_getcpu();
        #pragma omp critical
        {
            if(cpu_num > max_core)
                max_core = cpu_num;
            if(thread_num > max_thread)
                max_thread = thread_num;
        };
    }

    cout << "Threads used: " << max_thread + 1 << endl;
    cout << "Largest core used: " << max_core + 1 << " cores" << endl;*/

    size_t size = 1024*1024*128*8;
    vector<double> a(size);
    vector<double> b(size);
    vector<double> c(size);

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        a[i] = 0, b[i] = i, c[i] = size - i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        a[i] = b[i] + c[i];
    double t2 = omp_get_wtime();
    cout << "Linear BW: " << (size * sizeof(double) * 3)/((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        a[i] = b[i] + c[i];
    t2 = omp_get_wtime();
    cout << "Second linear BW: " << (size * sizeof(double) * 3)/((t2 - t1) * 1e9) << " GB/s" << endl;
}