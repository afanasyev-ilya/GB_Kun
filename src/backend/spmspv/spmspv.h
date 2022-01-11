#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bucket {
    VNT row;
    T val;
};

/*
vector<int> bucket_amount(_number_of_buckets);
vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));
vector<vector<bucket>> buckets(_number_of_buckets);
for (int i = 0; i < _number_of_buckets; i++) {
    buckets[i] = vector<bucket>(bucket_amount[i]);
}
vector<float> SPA(matrix_size);
vector<int> offset(_number_of_buckets);

 */

namespace lablas{
namespace backend{

template <typename T>
void SpMSpV(const Matrix<T> *_matrix,
            const SparseVector<T> *_x,
            Vector<T> *_y,
            Descriptor *_desc, int nb)
{

    VNT matrix_size;
    _matrix->get_csc()->get_size(&matrix_size);
    int nt = 32;
    auto v = (char *)calloc(1, sizeof(int) * (2*nb + nt * nb) + sizeof(float) * (matrix_size) + sizeof(bucket<T>) * (nb * nb));
    int memory_offset = 0;
    auto bucket_amount = (int *)(v + memory_offset);
    memory_offset += nb;
    auto offset_ = (int **)(v + memory_offset);
    memory_offset += nb * nt;
    auto buckets = (bucket<T> **)(v + memory_offset);
    memory_offset += nb;
    auto *SPA = (float *)(v + memory_offset);
    memory_offset += (int)matrix_size;
    int *offset = (int *)(v + memory_offset);
    memory_offset += nb;

//    cout << "SPMSPV x: ";
//    _x->print_storage_type();
//    cout << "SPMSPV y: ";
//    _y->print_storage_type();
    SpMSpV_csr((MatrixCSR<T> *) _matrix->get_csc(), _x, _y->getDense(), (int)nb, (int)nt,
               (int *)bucket_amount, (int **)offset_, (bucket<T> **)buckets, (float*)SPA, (int*)offset);

//    cout << "SPMSPV result: ";
//    _y->force_to_dense();
//    _y->print();
}

#include "spmspv_csr.h"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
