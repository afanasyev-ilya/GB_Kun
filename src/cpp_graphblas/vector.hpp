#pragma once

#include "../backend/vector/vector.h"
#include "types.hpp"
#include <vector>

namespace lablas {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Vector {
public:
    using ValueType = T;
public:
    Vector() : _vector() {}
    explicit Vector(Index nsize) : _vector(nsize) {}

    ~Vector() {}

    LA_Info build(const std::vector<Index>* indices,
                  const std::vector<T>*     values,
                  Index  nvals)
    {
        if (indices == NULL || values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(indices->data(), values->data(), nvals);
    }

    LA_Info build(const std::vector<T>*     values,
                  Index  nvals)
    {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values->data(), nvals);
    }

    LA_Info build(const T* values,
                  Index  nvals)
    {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values, nvals);
    }

    LA_Info fill(T val)
    {
        _vector.set_constant(val);
        return GrB_SUCCESS;
    }

    LA_Info clear()
    {
        return _vector.clear();
    }

    LA_Info set_element(T val, VNT pos)
    {
        _vector.set_element(val, pos);
        return GrB_SUCCESS;
    }

    inline backend::Vector<T>* get_vector()
    {
        return &(this->_vector);
    }

    LA_Info get_nvals(Index *_nvals) const
    {
        *_nvals = _vector.get_nvals();
        return GrB_SUCCESS;
    }

    const backend::Vector<T>* get_vector() const
    {
        return &_vector;
    }

    void print() const
    {
        return _vector.print();
    }

    void print_storage_type() const
    {
        _vector.print_storage_type();
    }

    // we can name vectors for debug purposes
    void set_name(const string &_name) {_vector.set_name(_name); };

    Index nvals() const { return _vector.get_nvals();};
    Index size() const {return _vector.get_size();};

    void swap(Vector *_another)
    {
        _vector.swap(_another->get_vector());
    }

    LA_Info fillAscending(Index nvals) {
        return _vector.fillAscending(nvals);
    }

    LA_Info dup (const Vector<T>* rhs) {
        return _vector.dup(rhs->get_vector());
    }

    T const & get_at(Index _index) const
    {
        return _vector.get_at(_index);
    }
private:
    backend::Vector<T> _vector;

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);

    template<typename Y>
    friend bool operator!=(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Vector>
class VectorIteratorType
{
public:
    using ValueType = typename Vector::ValueType;
public:
    VectorIteratorType(Vector& collection, Index const index) :
            index(index), collection(collection) {}

    bool operator!= (VectorIteratorType const & other) const
    {
        return index != other.index;
    }

    ValueType const & operator* () const
    {
        return collection.get_at(index);
    }

    VectorIteratorType const & operator++ ()
    {
        ++index;
        return *this;
    }

    VectorIteratorType const & operator-- ()
    {
        --index;
        return *this;
    }
private:
    Index   index;
    Vector&       collection;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
using VectorIterator = VectorIteratorType<Vector<T>>;

template <typename T>
using VectorConstIterator = VectorIteratorType<Vector<T>>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline VectorIterator<T> begin(Vector<T> &collection)
{
    return VectorIterator<T>(collection, 0);
}

template <typename T>
inline VectorIterator<T> end(Vector<T>& collection)
{
    return VectorIterator<T>(collection, collection.size());
}

template <typename T>
inline VectorConstIterator<T> begin(const Vector<T> &collection)
{
    return VectorConstIterator<T>(collection, 0);
}

template <typename T>
inline VectorConstIterator<T> end(const Vector<T>& collection)
{
    return VectorConstIterator<T>(collection, collection.size());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    return lhs._vector == rhs._vector;
}

template <typename T>
bool operator!=(Vector<T>& lhs, Vector<T>& rhs)
{
    return !(lhs._vector == rhs._vector);
}

template <typename T>
void print_diff(Vector<T>& lhs, Vector<T>& rhs)
{
    print_diff(lhs, rhs);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
