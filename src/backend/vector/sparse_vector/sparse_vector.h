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
        nvals = 0;
        MemoryAPI::allocate_array(&vals, size);
        MemoryAPI::allocate_array(&ids, size);
    };

    virtual ~SparseVector()
    {
        MemoryAPI::free_array(vals);
        MemoryAPI::free_array(ids);
    };

    void print() const
    {
        if(nvals == 0)
            cout << "vector is empty (from print)" << endl;
        else
        {
            for(VNT i = 0; i < nvals; i++)
            {
                cout << "["<< ids[i]<< "]:"  << vals[i] << " ";
            }
            cout << endl;
        }
    };

    void get_nvals(VNT* _nvals) const {
        *_nvals = nvals;
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

    LA_Info build(const Index* _indices, const T *_values, Index _nvals)
    {
        nvals = _nvals;
        if (nvals > size)
            return GrB_INDEX_OUT_OF_BOUNDS;

        for (Index i = 0; i < _nvals; i++){
            vals[i] = _values[i];
            ids[i] = _indices[i];
        }
        return GrB_SUCCESS;
    }

    LA_Info fillAscending(Index nvals) {
        throw "vector.fillAscending not implemented yet";
//        for (Index i = 0; i < nvals; i++)
//            vals[ids[i]] = i;
        /*NOT IMPLEMENTED YET*/

        return GrB_SUCCESS;
    }

    void dup(GenericVector<T>* rhs) {
        size = rhs->get_size();
        nvals = rhs->get_nvals();
        MemoryAPI::resize(&vals, size);
        MemoryAPI::resize(&ids, size);
        std::memcpy(vals,rhs->get_vals(), sizeof(T) * rhs->get_size());
        std::memcpy(ids,rhs->get_ids(), sizeof(T) * rhs->get_size());
    };

    bool isDense() const {
        return false;
    }
    bool isSparse() const {
        return true;
    }

    VNT get_nvals() const { return nvals; };
    void print_storage_type() const { cout << "It is sparse vector" << endl; };

    void set_element(T _val, VNT _pos);

    void set_all_constant(T _val);

    void fill_with_zeros() { nvals = 0; };

    void convert(DenseVector<T> *_dense_vector);

    VNT get_size() const {return size;};

    void set_size(VNT _nvals) { nvals = _nvals; };

    void clear() { nvals = 0;};
    void push_back(VNT _id, T _val) { ids[nvals] = _id; vals[nvals] = _val; nvals++; };
private:
    VNT size;
    ENT nvals;
    Storage get_storage() {return GrB_SPARSE; };

    VNT *ids;
    T *vals;
};

#include "sparse_vector.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
