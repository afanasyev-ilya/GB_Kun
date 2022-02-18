#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
struct bucket {
    VNT row;
    T val;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpV(const Matrix<T> *_matrix,
            const SparseVector<T> *_x,
            Vector<T> *_y,
            Descriptor *_desc,
            int nb)
{
    VNT matrix_size, nz;
    _matrix->get_csc()->get_size(&matrix_size);
    _x->get_nnz(&nz);
    //long long max_number_of_insertions = nz*_matrix->get_csc()->get_max_degree();
    long long max_number_of_insertions = nz*matrix_size;

    int nt = 32;
    //auto v = _matrix->get_workspace()->get_spmspv_buffer();
    auto v = (char *)calloc(1, sizeof(int) * (2*nb + nt * nb) + sizeof(float) * (matrix_size) + sizeof(bucket<T>) * (nb * max_number_of_insertions));
    int memory_offset = 0;
    auto bucket_amount = (int *)(v + memory_offset);
    memory_offset += nb * sizeof(int);
    auto offset_ = (int *)(v + memory_offset);
    memory_offset += nb * nt * sizeof(int);
    auto *SPA = (float *)(v + memory_offset);
    memory_offset += matrix_size * sizeof(float);
    int *offset = (int *)(v + memory_offset);
    memory_offset += nb * sizeof(int);
    auto buckets = (bucket<T> *)(v + memory_offset);
    //memory_offset += nb * max_number_of_insertions * sizeof(bucket<T>);

    SpMSpV_csr((MatrixCSR<T> *) _matrix->get_csc(), _x, _y->getDense(), (int)nb, (int)nt,
               (int *)bucket_amount, (int *)offset_, (bucket<T> *)buckets, (float*)SPA, (int*)offset);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpV_alloc(const Matrix<T> *_matrix,
                  const SparseVector<T> *_x,
                  Vector<T> *_y,
                  Descriptor *_desc,
                  int nb)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "spmspv_buckets.h"
#include "spmspv_csr.h"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
