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

}
}
