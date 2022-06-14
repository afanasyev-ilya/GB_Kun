#pragma once
/// @file vector.hpp
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief Vector class frontend wrapper for cpp usage
/// @details Describes GraphBLAS methods for vector object, invokes functions and methods of backend vector object
/// @date June 15, 2022

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
    /**
     * Create a new Vector object with default size
     * @brief Default constructor
     * @see Vector(Index nsize)
     */
    Vector() : _vector() {}

    /**
     * Create a new Vector object with given size
     * @brief Parameter constructor
     * @param nsize The maximum number of elements in a Vector
     * @see Vector()
     */
    explicit Vector(Index nsize) : _vector(nsize) {}

    ~Vector() {}

    /**
    * @brief Build Vector from STL vector of indices and appropriate values
    * @param[in] indices Pointer to the STL vector of indices which correspond to non-zero elements of a Vector
    * @param[in] values Pointer to the STL vector of values are to be placed according to corresponding indices
    * @param[in] nvals Total number of non-NULL elements in a Vector
    * @return la_info Flag for the correctness check
    */
    LA_Info build(const std::vector<Index>* indices,
                  const std::vector<T>*     values,
                  Index  nvals)
    {
        if (indices == NULL || values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(indices->data(), values->data(), nvals);
    }

    /**
    * @brief Build Vector from STL vector of values
    * @param[in] values Pointer to the STL vector of values are to be placed in the Vector
    * @param[in] nvals Total number of elements in a Vector
    * @return la_info Flag for the correctness check
    */
    LA_Info build(const std::vector<T>*     values,
                  Index  nvals)
    {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values->data(), nvals);
    }

    /**
    * @brief Build Vector from array of values
    * @param[in] values Pointer to the array of values are to be placed in the Vector
    * @param[in] nvals Total number of elements in a Vector
    * @return la_info Flag for the correctness check
    */
    LA_Info build(const T* values,
                  Index  nvals)
    {
        if (values == NULL) return GrB_NULL_POINTER;
        if (nvals == 0) return GrB_INVALID_VALUE;
        return _vector.build(values, nvals);
    }

    /**
    * @brief Fill Vector with the same value
    * @param[in] val Value to be placed in each element of the Vector
    * @return la_info Flag for the correctness check
    */
    LA_Info fill(T val)
    {
        _vector.set_constant(val);
        return GrB_SUCCESS;
    }

    /**
    * @brief Fill Vector with zeros
    * @return la_info Flag for the correctness check
    */
    LA_Info clear()
    {
        return _vector.clear();
    }

    /**
    * @brief Place a chosen element into a defined position
    * @param[in] val Value to be placed in the element of the Vector
    * @param[in] pos Position to be updated
    * @return la_info Flag for the correctness check
    */
    LA_Info set_element(T val, VNT pos)
    {
        _vector.set_element(val, pos);
        return GrB_SUCCESS;
    }

    /**
    * @brief Get a backend implementation of a Vector
    * @return vector Pointer to a backend vector
    */
    inline backend::Vector<T>* get_vector()
    {
        return &(this->_vector);
    }

    /**
    * @brief Get the number of non-zero elements is a vector
    * @param[out] _nvals Number of non-zero elements
    * @return la_info Flag for the correctness check
    */
    LA_Info get_nvals(Index *_nvals) const
    {
        *_nvals = _vector.get_nvals();
        return GrB_SUCCESS;
    }

    /**
    * @brief Get a backend implementation of a Vector
    * @return vector Const pointer to a backend vector
    */
    const backend::Vector<T>* get_vector() const
    {
        return &_vector;
    }

    /**
    * @brief Print Vector for debug purposes
    * @return Nothing
    */
    void print() const
    {
        return _vector.print();
    }

    /**
    * @brief Print main representation of a backend Vector (sparse or dense)
    * @return Nothing
    */
    void print_storage_type() const
    {
        _vector.print_storage_type();
    }

    /**
    * @brief Give Vector a name for debug purposes
    * @param[in] _name Name of a vector
    * @return Nothing
    */
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

    template<typename Y>
    friend void print_diff(Vector<Y>& lhs, Vector<Y>& rhs);
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
    print_diff(lhs._vector, rhs._vector);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
