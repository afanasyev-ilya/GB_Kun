#pragma once
#include <cstring>
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend{

template <typename T>
class SparseVector : public GenericVector<T>
{
public:
    SparseVector(int _size)
    {
        size = _size;
        nnz = 0;
        if(size == 0)
            cout << "!!!!!" << endl;
        MemoryAPI::allocate_array(&vals, size);
        MemoryAPI::allocate_array(&ids, size);
    };

    ~SparseVector()
    {
        MemoryAPI::free_array(vals);
        MemoryAPI::free_array(ids);
    };

    void print() const
    {
        if(nnz == 0)
            cout << "vector is empty (from print)" << endl;
        else
        {
            for(VNT i = 0; i < nnz; i++)
            {
                cout << "( "<< ids[i]<< " , "  << vals[i] << ") ";
            }
            cout << endl;
        }
    };

    void get_nnz(VNT* _nnz) const {
        *_nnz = nnz;
    }

    void get_size(VNT* _size) const {
        *_size = size;
    }

    T* get_vals () {
        return vals;
    }

    const T* get_vals () const {
        return vals;
    }

    VNT* get_ids () {
        return ids;
    }

    const VNT* get_ids () const {
        return ids;
    }

    LA_Info build(const Index* indices, const T *values, Index nvals) {

        if (nvals > size)
            return GrB_INDEX_OUT_OF_BOUNDS;

        for (Index i = 0; i < nvals; i++){
            vals[i] = values[i];
            ids[i] = indices[i];
        }
        return GrB_SUCCESS;
    }

    LA_Info fillAscending(Index nvals) {
//        for (Index i = 0; i < nvals; i++)
//            vals[ids[i]] = i;
        /*NOT IMPLEMENTED YET*/

        return GrB_SUCCESS;
    }

    void dup(GenericVector<T>* rhs) {
        if (rhs->isSparse()) {
            MemoryAPI::allocate_array(&vals, rhs->get_size());
            std::memcpy(vals,rhs->get_vals(), sizeof(T) * rhs->get_nvals());
        }
    };

    bool isDense() const {
        return false;
    }
    bool isSparse() const {
        return true;
    }

    VNT get_nvals() const { return nnz; };
    void print_storage_type() const { cout << "It is sparse vector" << endl; };

    void set_element(T _val, VNT _pos);

    void set_all_constant(T _val);

    void fill_with_zeros() { nnz = 0; };

    void convert(DenseVector<T> *_dense_vector);

    VNT get_size() const {return size;};
private:
    VNT size;
    ENT nnz;

    VNT *ids;
    T *vals;
};

#include "sparse_vector.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
