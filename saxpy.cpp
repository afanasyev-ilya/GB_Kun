#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define CMG_SIZE 12
#define MAX_CORES 48
#define CMG_NUM 4

//#include "fj_tool/fipp.h"

using namespace std;

void saxpy(float a, float * __restrict z, const float * __restrict x, const float * __restrict y, size_t size)
{
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        z[i] = a*x[i] + y[i];
}

void gather(const float *__restrict data, const int * __restrict indexes, float * __restrict result, size_t size)
{
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        result[i] = data[indexes[i]];
}

void gather_local_copy(const float *__restrict copy_data, const int * __restrict indexes, float * __restrict result, size_t size, int current_radius)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cmg = tid / CMG_SIZE;
        const float *__restrict local_data = &copy_data[current_radius * cmg];
        #pragma omp for
        for(size_t i = 0; i < size; i++)
            result[i] = local_data[indexes[i]];
    }
}

int main(void)
{
    size_t size = 1610612736; // 6.4 GB each
    float *x, *y, *z;
    x = (float*)memalign(0x200, size *sizeof(float));
    y = (float*)memalign(0x200, size *sizeof(float));
    z = (float*)memalign(0x200, size *sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 3.0f;
    }

    for(int i = 0; i < 4; i++) {
        double t1 = omp_get_wtime();
        //fipp_start();
        saxpy(2.0f, z, x, y, size);
        //fipp_stop();
        double t2 = omp_get_wtime();
        cout << size * sizeof(float) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;
    }
    free(x);
    free(y);
    free(z);

    float *result, *data;
    int *indexes;
    int large_size = size;
    int small_size = 1024*1024*8;

    result = (float*)memalign(0x200, large_size *sizeof(float));
    indexes = (int*)memalign(0x200, large_size *sizeof(int));
    data = (float*)memalign(0x200, small_size *sizeof(float));
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            result[i] = rand_r(&myseed);
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < small_size; i++)
        {
            data[i] = rand_r(&myseed)/RAND_MAX;
        }
    }
    const int num_tests = 6;
    int rads[num_tests] = {16*1024/sizeof(float),
                   32*1024/sizeof(float),
                   64*1024/sizeof(float),
                   256*1024/sizeof(float),
                   512*1024/sizeof(float),
                   1024*1024/sizeof(float)};

    cout << "large size is " << large_size * sizeof(int) / (1024*1024) << " MB" << endl;
    for(int idx = 0; idx < num_tests; idx ++)
    {
        int current_radius = rads[idx];
        cout << "Rad size is " << current_radius * sizeof(int) / (1024) << " KB" << endl;

        float *copy_data = (float*)memalign(0x200, CMG_NUM*small_size *sizeof(float));

        #pragma omp parallel
        {
            unsigned int myseed = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < large_size; i++)
            {
                indexes[i] = (int) rand_r(&myseed) % current_radius;
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int cmg = tid / CMG_SIZE;
            float *local_data = &copy_data[current_radius * cmg];
            if(tid % CMG_SIZE == 0)
            {
                for(int i = 0; i < current_radius; i++) // numa aware alloc
                {
                    local_data[i] = data[i];
                }
            }
        }

        double t1 = omp_get_wtime();
        gather(data, indexes, result, large_size);
        double t2 = omp_get_wtime();
        cout << current_radius * sizeof(int) / (1024) << "KB " << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;

        t1 = omp_get_wtime();
        gather_local_copy(copy_data, indexes, result, large_size, current_radius);
        t2 = omp_get_wtime();
        cout << current_radius * sizeof(int) / (1024) << "KB " << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s (copy data)" << endl;

        int threads = omp_get_max_threads();
        int seg_size = 64*1024;
        std::sort(indexes, indexes + large_size,
                  [seg_size](int a, int b) {return a/seg_size > b/seg_size; });

        t1 = omp_get_wtime();
        gather_local_copy(copy_data, indexes, result, large_size, current_radius);
        t2 = omp_get_wtime();
        cout << current_radius * sizeof(int) / (1024) << "KB " << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s (copy data, sorted)" << endl;
        cout << endl;

        free(copy_data);
    }

    free(data);
    free(indexes);
    free(result);
}
