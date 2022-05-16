//#include <tbb/parallel_for.h>
#include "src/gb_kun.h"

/*struct mytask
{
    mytask(size_t n)
            :_n(n)
    {}
    void operator()() {
        for (int i = 0;i < 1000000; ++i) {}  // Deliberately run slow
        std::cerr << "[" << _n << "]";
    }
    size_t _n;
};

int main(int,char**)
{
    std::vector<int> a(100000000, 0);
    std::vector<int> b(100000000, 0);
    std::vector<int> c(100000000, 0);


    for(auto &it: a)
    {
        int pos = &it - &a[0];
        it = pos;
    }
    for(auto &it: b)
    {
        int pos = &it - &b[0];
        it = b.size() - pos;
    }

    {
        Timer tm("first for");
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0,a.size()),
                [&](const tbb::blocked_range<size_t>& r)
                {
                    for (size_t i=r.begin();i<r.end();++i)
                        c[i] = b[i] + a[i];
                }
        , tbb::static_partitioner());
        std::cout << "BW " << 3.0*a.size()*sizeof(int)/(tm.get_time_ms()*1e6) << " GB/s" << std::endl;
    }

    {
        Timer tm("second for");
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0,a.size()),
                [&](const tbb::blocked_range<size_t>& r)
                {
                    for (size_t i=r.begin();i<r.end();++i)
                        c[i] = b[i] + a[i];
                }
        , tbb::static_partitioner());
        std::cout << "BW " << 3.0*a.size()*sizeof(int)/(tm.get_time_ms()*1e6) << " GB/s" << std::endl;
    }

    {
        Timer tm("third for");
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0,a.size()),
                [&](const tbb::blocked_range<size_t>& r)
                {
                    for (size_t i=r.begin();i<r.end();++i)
                        c[i] = b[i] + a[i];
                }
                , tbb::static_partitioner());
        std::cout << "BW " << 3.0*a.size()*sizeof(int)/(tm.get_time_ms()*1e6) << " GB/s" << std::endl;
    }

    return 0;
}*/

template <typename T>
void numa_aware_alloc(T **_ptr, size_t _size, int _target_socket)
{
    *_ptr = (T*)malloc(_size*sizeof(T)); // malloc uses first-touch policy for memory allocations

    #ifdef __USE_KUNPENG__ // currently only Kunpeng platform has 2 sockets, where numa-aware malloc makes sense
    const int threads_per_socket = sysconf(_SC_NPROCESSORS_ONLN)/2;
    int threads_active_on_target_socket = 0;
        #pragma omp parallel
    {
        int cur_cpu = sched_getcpu();
        int cur_socket = cur_cpu / threads_per_socket;

        if(cur_socket == _target_socket)
        {
            #pragma omp atomic
            threads_active_on_target_socket += 1;
            // we need to consider situations, when for example 10 threads run
            // on first socket, while 2 -- on second
        }

        #pragma omp barrier // wait for all atomics to finish

        size_t work_per_thread = (_size - 1)/threads_active_on_target_socket + 1;
        if(cur_socket == _target_socket) // init target array using threads only from target socket
        {
            int tid = omp_get_thread_num() % threads_active_on_target_socket;
            for(size_t i = tid*work_per_thread; i < min((tid+1)*work_per_thread, _size); i++)
            {
                (*_ptr)[i] = 0;
            }
        }
    }
    #else
    #pragma omp parallel for
    for(size_t i = 0; i < _size; i++)
    {
        (*_ptr)[i] = 0;
    }
    #endif
}


int main()
{
    int *a;
    int size = 1024;
    numa_aware_alloc(&a, size, 0);
    for(int i = 0; i < size; i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    int *b;
    numa_aware_alloc(&b, size, 1);
    for(int i = 0; i < size; i++)
    {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}