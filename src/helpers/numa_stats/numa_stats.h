/**
  @file numa_stats.h
  @author S.krymskiy
  @version Revision 1.1
  @date June 10, 2022
*/

#pragma once

#include <unistd.h>

#ifdef __USE_KUNPENG__
int numCPU() { return sysconf(_SC_NPROCESSORS_ONLN); };
#else
int numCPU() { return 1; };
#endif

#ifdef __USE_KUNPENG__
int cores_per_socket()
{
    return sysconf(_SC_NPROCESSORS_ONLN)/2;
};
#else
int cores_per_socket()
{
    return omp_get_max_threads();
};
#endif

/**
 * num_sockets_used function.
 * @brief returns number of sockets used in parallel region
*/

int num_sockets_used()
{
    #ifdef __USE_KUNPENG__
    const int num_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    int threads_amount;
    int cpu[num_cpu];
    #pragma omp parallel
    {
        threads_amount = omp_get_num_threads();
        int cpu_num = sched_getcpu();
        cpu[omp_get_thread_num()] = cpu_num;
    }
    bool socket[2];
    socket[0] = false;
    socket[1] = false;
    for (int i = 0; i < threads_amount && !(socket[0] && socket[1]); i++)
    {
        if (cpu[i] < num_cpu / 2)
        {
            socket[0] = true;
        }
        else
        {
            socket[1] = true;
        }
    }
    if(socket[0] && socket[1])
        return 2;
    else
        return 1;
    #else
    return 1;
    #endif
}