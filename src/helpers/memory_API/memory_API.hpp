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

    /*#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < _size; i++)
        (*_ptr)[i] = 0;*/
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
