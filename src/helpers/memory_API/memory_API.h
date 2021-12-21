#pragma once

#ifdef __USE_KNL__
#include <mm_malloc.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MemoryAPI
{
public:
    template <typename T>
    static void allocate_array(T **_ptr, size_t _size);

    template <typename T>
    static void allocate_host_array(T **_ptr, size_t _size);

    template<typename T>
    static void numa_aware_alloc(T **_ptr, size_t _size, int _target_socket);

    template <typename T>
    static void free_array(T *_ptr);

    template <typename T>
    static void free_host_array(T *_ptr);

    template <typename T>
    static void copy(T *_dst, const T *_src, size_t _size);

    template <typename T>
    static void set(T *_data, T val, size_t _size);

    template <typename T>
    static void resize(T **_ptr, size_t _new_size);

    #ifdef __USE_GPU__
    template <typename T>
    static void move_array_to_device(T *_ptr, size_t _size);

    template <typename T>
    static void move_array_to_host(T *_ptr, size_t _size);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "memory_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

