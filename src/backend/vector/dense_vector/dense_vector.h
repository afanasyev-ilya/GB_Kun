#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
class DenseVector : public GenericVector<T>
{
public:
    DenseVector(VNT _size);
    ~DenseVector();

    void print() const;

    void get_size (VNT* _size) const {
        *_size = size;
    }

    VNT get_size () const {
        return size;
    }

    T* get_vals () {
        return vals;
    }

    const T* get_vals () const {
        return vals;
    }

    LA_Info build(const T* values,
                  VNT nvals) {
        if (nvals > size){
            return GrB_INDEX_OUT_OF_BOUNDS;
        }
        #pragma omp parallel for schedule(static)
        for (VNT i = 0; i < nvals; i++) {
            vals[i] = values[i];
        }
        return GrB_SUCCESS;
    }

    VNT get_nvals() const;

    void print_storage_type() const { cout << "It is dense vector" << endl; };

    void set_element(T _val, VNT _pos);
    void set_all_constant(T _val);

    void fill_with_zeros();
private:
    VNT size;
    T *vals;

    template<typename Y>
    friend bool operator==(DenseVector<Y>& lhs, DenseVector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "dense_vector.hpp"

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
