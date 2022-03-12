#pragma once

#include <stdio.h>
#include <sched.h>
#ifdef __USE_KUNPENG__
#include <omp.h>
#endif

#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)       \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
/*printf("matrix has %ld\n edges", nvals);*/                                            \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                           \
double my_perf = my_nvals * 2.0 / ((my_t2 - my_t1)*1e9);                                         \
double my_bw = my_nvals * bytes_per_flop/((my_t2 - my_t1)*1e9);                                  \
/*printf("edges: %lf\n", nvals);*/                                                   \
/*printf("%s time %lf (ms)\n", op_name, (my_t2-my_t1)*1000); */                          \
/*printf("%s perf %lf (GFLop/s)\n", op_name, perf);*/                                   \
printf("%s time = %lf (ms)\n", op_name, my_time);                                      \
printf("%s BW = %lf (GB/s)\n", op_name, my_bw);                                      \
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");                                                 \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);                                                                           \

#define SAVE_TIME(call_instruction, op_name)       \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                           \
double my_perf = 0;                                        \
double my_bw = 0;                                  \
size_t my_nvals = 0;                               \
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");                                                 \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);                                                                           \


#define SAVE_TIME_SEC(call_instruction, op_name)       \
my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
my_t2 = omp_get_wtime();                                                         \
my_time = (my_t2 - my_t1)*1000;                                                           \
my_perf = 0;                                        \
my_bw = 0;                                  \
my_nvals = 0;                               \
my_f = fopen("perf_stats.txt", "a");                                                 \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);

#define SAVE_TEPS(call_instruction, op_name, iterations, matrix)                        \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                  \
double my_perf = iterations*(my_nvals / ((my_t2 - my_t1)*1e6));                         \
double my_bw = 0;                                                                       \
FILE *my_f;                                                                             \
my_f = fopen("perf_stats.txt", "a");                                                    \
fprintf(my_f, "%s %lf (ms) %lf (MTEPS/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f);                                                                           \

void print_omp_stats()
{
    #ifdef __USE_KUNPENG__
    #pragma omp parallel
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
    cout << "Largest core used: " << max_core + 1 << " cores" << endl;
    #endif

    /*size_t size = 1024*1024*128*8;
    double *a, *b, *c;
    MemoryAPI::allocate_array(&a, size);
    MemoryAPI::allocate_array(&b, size);
    MemoryAPI::allocate_array(&c, size);

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

    MemoryAPI::free_array(a);
    MemoryAPI::free_array(b);
    MemoryAPI::free_array(c);*/
}
