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
        Vector(int _size): dense(_size), sparse(_size) {size = _size; nz = size;};
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

    private:
        VNT size;
        VNT nz;
        DenseVector<T> dense;
        SparseVector<T> sparse;

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


