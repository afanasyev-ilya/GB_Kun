#pragma once

namespace lablas {
namespace backend {

class Workspace {
public:
    Workspace(Index _nrows, Index _ncols, Index max_nz_in_col)
    {
        Index vector_size = max(_nrows, _ncols);
        MemoryAPI::allocate_array(&mask_conversion, vector_size);
        MemoryAPI::numa_aware_alloc(&first_socket_vector, vector_size, 0);
        MemoryAPI::numa_aware_alloc(&second_socket_vector, vector_size, 1);

        MemoryAPI::numa_aware_alloc(&prefetched_vector, vector_size, 0); // TODO maybe on both sockets

        int nb = 512;
        int nt = omp_get_max_threads();

        size_t max_number_of_insertions = (vector_size) * max_nz_in_col;
        size_t spmspv_buffer_size = sizeof(int) * (2*nb + nt * nb) + sizeof(float) * (vector_size) + (sizeof(double) + sizeof(VNT)) * (nb * max_number_of_insertions);
        cout << spmspv_buffer_size / 1e6 << " MB" << endl;
        MemoryAPI::allocate_array(&spmspv_buffer, spmspv_buffer_size);
    }

    ~Workspace()
    {
        MemoryAPI::free_array(mask_conversion);
        MemoryAPI::free_array(first_socket_vector);
        MemoryAPI::free_array(second_socket_vector);
        MemoryAPI::free_array(prefetched_vector);
        MemoryAPI::free_array(spmspv_buffer);
    }

    Index *get_mask_conversion() { return mask_conversion; };

    double *get_first_socket_vector() { return first_socket_vector; };

    double *get_second_socket_vector() { return second_socket_vector; };

    double *get_prefetched_vector() { return prefetched_vector; };

    char *get_spmspv_buffer() {return spmspv_buffer;};
private:
    Index *mask_conversion;
    double *first_socket_vector;
    double *second_socket_vector;
    double *prefetched_vector;

    char *spmspv_buffer;
};

}
}