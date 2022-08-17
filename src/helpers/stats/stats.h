/**
  @file stats.h
  @author S.krymskiy
  @version Revision 1.1
  @date June 10, 2022
*/

#pragma once

#include <stdio.h>
#ifdef __USE_KUNPENG__
#include <sched.h>
#endif
#include <omp.h>
#include <unistd.h>

double SPMV_TIME;
double SPMSPV_TIME;
double CONVERT_TIME;

#ifdef __DEBUG_INFO__
#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)       \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                  \
double my_perf = my_nvals * 2.0 / ((my_t2 - my_t1)*1e9);                                \
double my_bw = my_nvals * bytes_per_flop/((my_t2 - my_t1)*1e9);                         \
printf("%s time = %lf (ms)\n", op_name, my_time);                                      \
printf("%s BW = %lf (GB/s)\n", op_name, my_bw);                                      \
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");                                                    \
if (my_time) {\
    fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals); \
}                                                                                        \
fclose(my_f);
#else
#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)       \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                  \
double my_perf = my_nvals * 2.0 / ((my_t2 - my_t1)*1e9);                                \
double my_bw = my_nvals * bytes_per_flop/((my_t2 - my_t1)*1e9);                         \
FILE *my_f;                                                                             \
my_f = fopen("perf_stats.txt", "a");                                                    \
if (my_time) {                                                                          \
    fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals); \
}                                                                                       \
fclose(my_f);
#endif

#define SAVE_TIME(call_instruction, op_name)       \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                           \
double my_perf = 0;                                        \
double my_bw = 0;                                  \
size_t my_nvals = 0;                               \
FILE *my_f;                                                                          \
my_f = fopen("perf_stats.txt", "a");               \
if (my_time) {\
    fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals); \
}                                                   \
fclose(my_f);                                                                           \

#define SAVE_TIME_SEC(call_instruction, op_name)       \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                       \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1);                                                           \
double my_perf = 0;                                        \
double my_bw = 0;                                  \
size_t my_nvals = 0;                               \
FILE *my_f = fopen("perf_stats.txt", "a");             \
if (my_time) {\
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals); \
}                                                       \
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
if (my_time) {\
    fprintf(my_f, "%s %lf (ms) %lf (MTEPS/s) %lf (GB/s) %lld\n", op_name, my_time, my_perf, my_bw, my_nvals); \
}                                                                                        \
fclose(my_f);                                                                           \

/**
 * save_teps function.
 * @brief saves performance statistics for an operation
 * @param _op_name name of the operation
 * @param _time time in ms
 * @param _nvals number of elements (for example matrix elements) processed in the region
 * @param _iterations amount of iterations
*/

void save_teps(const char *_op_name, double _time /*in ms*/, size_t _nvals, int _iterations = 1)
{
    double my_time = _time;
    double my_perf = _iterations*(_nvals / (_time*1e3)); // 1e3 instead of 1e6 since time in ms
    double my_bw = 0;
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    if (my_time) {
        fprintf(my_f, "%s %lf (ms) %lf (MTEPS/s) %lf (GB/s) %ld\n", _op_name, my_time, my_perf, my_bw, _nvals);
    }
    fclose(my_f);
}

/**
 * save_time_in_ms function.
 * @brief outputs time taken by the _op_name operation in ms
 * @param _op_name name of the operation
 * @param _time time in seconds
*/

void save_time_in_ms(const char *_op_name, double _time)
{
    double my_t2 = omp_get_wtime();
    double my_time = (_time)*1000;
    double my_perf = 0;
    double my_bw = 0;
    size_t my_nvals = 0;
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    if (my_time) {
        fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", _op_name, my_time, my_perf, my_bw, my_nvals);
    }
    fclose(my_f);
}

/**
 * save_time_in_sec function.
 * @brief outputs time taken by the _op_name operation in seconds
 * @param _op_name name of the operation
 * @param _time time in seconds
*/

void save_time_in_sec(const char *_op_name, double _time)
{
    double my_t2 = omp_get_wtime();
    double my_time = (_time);
    double my_perf = 0;
    double my_bw = 0;
    size_t my_nvals = 0;
    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    if (my_time) {
        fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", _op_name, my_time, my_perf, my_bw, my_nvals);
    }
    fclose(my_f);
}

/**
 * print_omp_stats function.
 * @brief outputs omp stats (Amount of threads, largest core, the number of a core for each thread).
*/

void print_omp_stats()
{
    #ifdef __DEBUG_INFO__
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
