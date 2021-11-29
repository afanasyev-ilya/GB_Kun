#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "dense_vector/dense_vector.h"
#include "sparse_vector/sparse_vector.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Vector {
public:
    Vector(int _size): dense(_size), sparse(_size) {size = _size; nz = size;};
    ~Vector(){};

    void set_constant(T _val) {
        dense.set_constant(_val);
        sparse.set_constant(_val);
    };
private:
    VNT size;
    VNT nz;
    DenseVector<T> dense;
    SparseVector<T> sparse;

    template<typename Y>
    friend void SpMV(Matrix<Y> &_matrix,
                     Vector<Y> &_x,
                     Vector<Y> &_y,
                     Descriptor &_desc);

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    return lhs.dense == rhs.dense;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


