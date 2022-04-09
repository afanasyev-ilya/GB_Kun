#pragma once

namespace lablas {
namespace backend {

class Workspace {
public:
    Workspace(Index _nrows, Index _ncols)
    {
        Index vector_size = max(_nrows, _ncols);
        MemoryAPI::allocate_array(&mask_conversion, vector_size);
        MemoryAPI::numa_aware_alloc(&first_socket_vector, vector_size, 0);
        MemoryAPI::numa_aware_alloc(&second_socket_vector, vector_size, 1);

        MemoryAPI::allocate_array(&shared_one_vector, vector_size);
    }

    ~Workspace()
    {
        MemoryAPI::free_array(mask_conversion);
        MemoryAPI::free_array(first_socket_vector);
        MemoryAPI::free_array(second_socket_vector);
        MemoryAPI::free_array(shared_one_vector);
    }

    Index *get_mask_conversion() { return mask_conversion; };

    double *get_first_socket_vector() { return first_socket_vector; };

    double *get_second_socket_vector() { return second_socket_vector; };

    double *get_shared_one() { return shared_one_vector; };

    char *get_spmspv_buffer() {return spmspv_buffer;};
private:
    Index *mask_conversion;
    double *first_socket_vector;
    double *second_socket_vector;
    double *shared_one_vector;

    char *spmspv_buffer;
};

/*In order to divide vector between two sockets */
template <typename T>
class ReducedWorkspace {
public:
    explicit ReducedWorkspace(Index size)
    {
        auto max_threads = omp_get_max_threads();
        if (max_threads == numCPU() * 2) {
            one_socket = false;
            threshold = (size / 2) + (size % 2);
            MemoryAPI::numa_aware_alloc(&first_socket_vector, threshold, 0);
            MemoryAPI::numa_aware_alloc(&second_socket_vector, size - threshold, 1);
        }
        if (max_threads == numCPU()) {
            one_socket = true;
            MemoryAPI::numa_aware_alloc(&first_socket_vector, size, 0);
        }
        /* for local debug */
        if (max_threads < numCPU()) {
            one_socket = true;
            MemoryAPI::allocate_array_new(&first_socket_vector, size);
        }
    }

    ~ReducedWorkspace()
    {
        MemoryAPI::free_array_new(first_socket_vector);
        if (!one_socket) {
            MemoryAPI::free_array_new(second_socket_vector);
        }
    }
    T *get_first_socket_vector() { return first_socket_vector; };

    T *get_second_socket_vector() { return second_socket_vector; };

    inline T& get_element(Index i) {
        if (one_socket) {
            return first_socket_vector[i];
        }
        if (!one_socket) {
            if (i < threshold) {
                return first_socket_vector[i];
            } else {
                return second_socket_vector[i - threshold];
            }
        }
        return first_socket_vector[i];
    }

private:
    T *first_socket_vector;
    T *second_socket_vector;
    bool one_socket;
    Index threshold;
};


template <typename T>
class CommonWorkspace {
public:
    explicit CommonWorkspace(Index size, T* vals)
    {
        auto max_threads = omp_get_max_threads();
        if (max_threads > numCPU()) {
            MemoryAPI::numa_aware_alloc_valued(&first_socket_vector, size, 0, vals);
            MemoryAPI::numa_aware_alloc_valued(&second_socket_vector, size, 1, vals);
            one_socket = false;
        }
        if (max_threads == numCPU()) {
            std::cout << "HERE_2: " <<  numCPU() << std::endl;
            one_socket = true;
            MemoryAPI::numa_aware_alloc_valued(&first_socket_vector, size, 0, vals);
        }
        /* for local debug */
        if (max_threads < numCPU()) {
            std::cout << "HERE_1" << std::endl;
            one_socket = true;
            MemoryAPI::numa_aware_alloc_valued(&first_socket_vector, size, 0, vals);
        }
    }

    ~CommonWorkspace()
    {
        MemoryAPI::free_array(first_socket_vector);
        if (!one_socket) {
            MemoryAPI::free_array(second_socket_vector);
        }
    }

    T *get_first_socket_vector() { return first_socket_vector; };

    T *get_second_socket_vector() { return second_socket_vector; };

//    inline T& get_element(Index i) {
//        if (one_socket) {
//            return first_socket_vector[i];
//        }
//        if (!one_socket) {
//            if (i < threshold) {
//                return first_socket_vector[i];
//            } else {
//                return second_socket_vector[i - threshold];
//            }
//        }
//        return first_socket_vector[i];
//    }

private:
    T *first_socket_vector;
    T *second_socket_vector;
    bool one_socket;
};

}
}
