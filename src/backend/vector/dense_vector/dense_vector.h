#pragma once
#include <cstring>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
class SparseVector;

template <typename T>
class DenseVector : public GenericVector<T>
{
public:
    DenseVector(VNT _size);
    virtual ~DenseVector();

    void print() const;

    T* get_vals () {
        return vals;
    }

    const T* get_vals () const {
        return vals;
    }

    VNT* get_ids () {
        return nullptr;
    }

    const VNT* get_ids () const {
        return nullptr;
    }

    LA_Info build(const T* values,
                  VNT nvals)
    {
        if (nvals > size)
        {
            return GrB_INDEX_OUT_OF_BOUNDS;
        }
        #pragma omp parallel for schedule(static)
        for (VNT i = 0; i < nvals; i++)
        {
            vals[i] = values[i];
        }
        return GrB_SUCCESS;
    }

    VNT get_nvals() const;

    void print_storage_type() const { cout << "It is a dense vector" << endl; };

    void set_element(T _val, VNT _pos);
    void set_all_constant(T _val);

    void fill_with_zeros();

    void convert(SparseVector<T> *_sparse_vector);

    LA_Info fillAscending(Index nvals) {
        #pragma omp parallel for
        for (Index i = 0; i < nvals; i++)
            vals[i] = i;

        return GrB_SUCCESS;
    }

    bool isDense() const {
        return true;
    }
    bool isSparse() const {
        return false;
    }

    void dup(GenericVector<T>* rhs) {
        size = rhs->get_size();
        MemoryAPI::resize(&vals, size);
        std::memcpy(vals, rhs->get_vals(), sizeof(T) * rhs->get_size());
    };

    Storage get_storage() {return GrB_DENSE; };

    VNT get_size() const {return size;};
private:
    VNT size;
    T *vals;

    template<typename Y>
    friend bool operator==(DenseVector<Y>& lhs, DenseVector<Y>& rhs);
    template<typename Y>
    friend bool operator!=(DenseVector<Y>& lhs, DenseVector<Y>& rhs);
    template<typename Y>
    friend void print_diff(DenseVector<Y>& lhs, DenseVector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(DenseVector<T>& lhs, DenseVector<T>& rhs)
{
    if(lhs.size != rhs.size)
        return false;

    for(VNT i = 0; i < lhs.size; i++)
    {
        if(fabs(lhs.vals[i] - rhs.vals[i]) > 0.001)
        {
            return false;
        }
    }

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator!=(DenseVector<T>& lhs, DenseVector<T>& rhs)
{
    return !(lhs == rhs);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void print_diff(DenseVector<T>& lhs, DenseVector<T>& rhs)
{
    VNT error_count = 0;
    for(VNT i = 0; i < lhs.size; i++)
    {
        if(fabs(lhs.vals[i] - rhs.vals[i]) > 0.001)
        {
            if(error_count < 10)
                cout << "Error in " << i << " : " << lhs.vals[i] << " " << rhs.vals[i] << endl;
            error_count++;
        }
    }

    for(VNT i = 0; i < min(lhs.size, (VNT)CHECK_PRINT_NUM); i++)
    {
        cout << "check " << i << " : " << lhs.vals[i] << " vs " << rhs.vals[i] << endl;
    }

    cout << "error_count: " << error_count << "/" << max(lhs.size, rhs.size)  << endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "dense_vector.hpp"

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
