#include <tbb/parallel_for.h>
#include <iostream>
#include <vector>
#include "src/helpers/timer/timer.h"

struct mytask
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
}