#pragma once

class Descriptor
{
public:
    Descriptor(VNT _matrix_size)
    {
        MemoryAPI::allocate_array(&tmp_buffer, _matrix_size);
    }
    ~Descriptor()
    {
        MemoryAPI::free_array(tmp_buffer);
    }
private:
    double *tmp_buffer;

    template <typename T>
    friend void SpMV(MatrixCSR<T> &_matrix,
                     MatrixCSR<T> &_matrix_socket_dub,
                     DenseVector<T> &_x,
                     DenseVector<T> &_y,
                     Descriptor &_desc);
};