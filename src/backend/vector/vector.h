#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "dense_vector/dense_vector.h"
#include "sparse_vector/sparse_vector.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas {
namespace backend {

template<typename T>
class Vector {
public:
    Vector(int _size): dense(_size), sparse(_size), storage(GrB_DENSE) {size = _size; nz = size;};
    ~Vector(){};

    void set_constant(T _val) {dense.set_constant(_val);};

    DenseVector<T>* getDense() {
        return &dense;
    }
    SparseVector<T>* getSparse() {
        return &sparse;
    }
    const DenseVector<T>* getDense() const {
        return &dense;
    }
    const SparseVector<T>* getSparse() const {
        return &sparse;
    }
    void getStorage(Storage* _storage) const{
        *_storage = storage;
    }
    void setStorage(Storage _storage) {
        storage = _storage;
    }
    void set_element(T val, VNT pos) {
        if (storage == GrB_DENSE) {
            dense.get_vals()[pos] = val;
        } else if (storage == GrB_SPARSE) {
            /* we count pos in NZ numbers, or in SIZE? */
            sparse.get_vals()[pos] = val;
        }
    }

    LA_Info build (const Index* indices,
                   const T*     values,
                   Index nvals) {
        storage = GrB_SPARSE;
        return sparse.build(indices, values, nvals);
    }

    LA_Info build(const T*    values,
                  Index nvals) {
        storage = GrB_DENSE;
        return dense.build(values, nvals);
    }

    void print() const {
        if (storage == GrB_DENSE) {
            return dense.print();
        }
        if (storage == GrB_SPARSE) {
            return sparse.print();
        }
    }

    void swap(Vector* rhs) {
        if (storage == GrB_DENSE) {
           dense.swap(rhs->getDense());
        }
        if (storage == GrB_SPARSE) {
            sparse.swap(rhs->getSparse());
        }
    }

    VNT nvals() {return nz;};
private:
    VNT size;
    VNT nz;
    DenseVector<T> dense;
    SparseVector<T> sparse;
    Storage storage;

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    return lhs.dense == rhs.dense;
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


