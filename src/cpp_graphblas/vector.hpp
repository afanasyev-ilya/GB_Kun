#ifndef GB_KUN_VECTOR_HPP
#define GB_KUN_VECTOR_HPP

#include "../backend/vector/vector.h"
#include "../helpers/utils.hpp"
#include "types.hpp"
#include <vector>

namespace lablas {

template<typename T>
class Vector {
public:
    Vector() : _vector() {}
    explicit Vector(Index nsize) : _vector(nsize) {}

    ~Vector() {}

    LA_Info build(const std::vector<Index>* indices,
                  const std::vector<T>*     values,
               Index  nvals) {

        if (indices == NULL || values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(indices->data(), values->data(), nvals);
    }

    LA_Info build(const std::vector<T>*     values,
                  Index  nvals) {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values->data(), nvals);
    }

    LA_Info build(const T* values,
                  Index  nvals) {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values, nvals);
    }

    LA_Info fill(T val) {
        _vector.set_constant(val);
        return GrB_SUCCESS;
    }

    LA_Info set_element(T val, VNT pos) {
        _vector.set_element(val, pos);
        return GrB_SUCCESS;
    }

    backend::Vector<T>* get_vector() {
        return &_vector;
    }

    const backend::Vector<T>* get_vector() const {
        return &_vector;
    }

    void print() const {
        return _vector.print();
    }

    void swap(Vector* rhs) {
        return _vector.swap(rhs->get_vector());
    }

    Index nvals() const { return _vector.nvals();};
private:
    backend::Vector<T> _vector;

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    return lhs._vector == rhs._vector;
}

}
#endif //GB_KUN_VECTOR_HPP
