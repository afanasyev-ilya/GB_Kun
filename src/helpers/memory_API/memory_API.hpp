/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::allocate_array(T **_ptr, size_t _size)
{
    #if defined(__USE_NEC_SX_AURORA__)
    *_ptr = (T*)aligned_alloc(sizeof(T), _size*sizeof(T));
    #elif defined(__USE_GPU__)
    SAFE_CALL(cudaMallocManaged((void**)_ptr, _size * sizeof(T)));
    #elif defined(__USE_KNL__)
    *_ptr = (T*)_mm_malloc(sizeof(T)*(_size),2097152);
    #else
    *_ptr = (T*)malloc(_size*sizeof(T));
    #endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::allocate_array_new(T **_ptr, size_t _size)
{
#if defined(__USE_NEC_SX_AURORA__)
    *_ptr = (T*)aligned_alloc(sizeof(T), _size*sizeof(T));
#elif defined(__USE_GPU__)
    SAFE_CALL(cudaMallocManaged((void**)_ptr, _size * sizeof(T)));
#elif defined(__USE_KNL__)
    *_ptr = (T*)_mm_malloc(sizeof(T)*(_size),2097152);
#else
    *_ptr = new T[_size]();
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::numa_aware_alloc(T **_ptr, size_t _size, int _target_socket)
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::numa_aware_alloc_valued(T **_ptr, size_t _size, int _target_socket, T* vals)
{
    *_ptr = (T*)malloc(_size*sizeof(T));

    int max_threads = omp_get_max_threads();
    if(max_threads == 2*numCPU())
    {
        #pragma omp parallel num_threads(2*numCPU())
        {
            size_t sock = omp_get_thread_num() / numCPU();
            size_t tid = omp_get_thread_num() % numCPU();

            size_t work_per_thread = (_size - 1)/numCPU() + 1;
            if(sock == _target_socket)
            {

                for(size_t i = tid*work_per_thread; i < min((tid+1)*work_per_thread, _size); i++)
                {
                    (*_ptr)[i] = vals[i];
                }
            }
        }
    }
    else if(omp_get_max_threads() == numCPU())
    {
#pragma omp parallel for num_threads(numCPU())
        for(size_t i = 0; i < _size; i++)
        {
            (*_ptr)[i] = vals[i];
        }
    }
    else
    {
#pragma omp parallel for
        for(size_t i = 0; i < _size; i++)
        {
            (*_ptr)[i] = vals[i];
        }
    }


}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::allocate_host_array(T **_ptr, size_t _size)
{
    *_ptr = (T*)malloc(_size*sizeof(T));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::free_array(T *_ptr)
{
    if(_ptr != NULL)
    {
        #if defined(__USE_NEC_SX_AURORA__)
        free(_ptr);
        #elif defined(__USE_GPU__)
        SAFE_CALL(cudaFree((void*)_ptr));
        #elif defined(__USE_KNL__)
        free(_ptr);
        #else
        free(_ptr);
        #endif
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::free_array_new(T *_ptr)
{
    if(_ptr != NULL)
    {
#if defined(__USE_NEC_SX_AURORA__)
        free(_ptr);
#elif defined(__USE_GPU__)
        SAFE_CALL(cudaFree((void*)_ptr));
#elif defined(__USE_KNL__)
        free(_ptr);
#else
        delete[] _ptr;
#endif
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::free_host_array(T *_ptr)
{
    if(_ptr != NULL)
    {
        free(_ptr);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::copy(T *_dst, const T *_src, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel
    for(long long i = 0; i < _size; i++)
    {
        _dst[i] = _src[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::set(T *_data, T _val, size_t _size)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < _size; i++)
    {
        _data[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MemoryAPI::resize(T **_ptr, size_t _new_size)
{
    if(*_ptr != NULL)
        MemoryAPI::free_array(*_ptr);
    MemoryAPI::allocate_array(_ptr, _new_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename T>
void MemoryAPI::move_array_to_device(T *_ptr, size_t _size)
{
    int device_id = 0;
    SAFE_CALL(cudaGetDevice(&device_id));
    SAFE_CALL(cudaMemPrefetchAsync(_ptr, _size*sizeof(T), device_id, NULL));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename T>
void MemoryAPI::move_array_to_host(T *_ptr, size_t _size)
{
    SAFE_CALL(cudaMemPrefetchAsync(_ptr, _size*sizeof(T), cudaCpuDeviceId, NULL));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
