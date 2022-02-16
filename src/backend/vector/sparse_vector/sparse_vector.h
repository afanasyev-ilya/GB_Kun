#pragma once

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

    ~SparseVector()
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
                cout << "( "<< ids[i]<< " , "  << vals[i] << ") ";
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

    Storage get_storage() {return GrB_SPARSE; };

    LA_Info build(const Index* indices, const T *values, Index nvals) {

        if (nvals > size)
            return GrB_INDEX_OUT_OF_BOUNDS;

        for (Index i = 0; i < nvals; i++){
            vals[i] = values[i];
            ids[i] = indices[i];
        }
        return GrB_SUCCESS;
    }

    VNT get_nvals() const { return nvals; };
    void print_storage_type() const { cout << "It is sparse vector" << endl; };

    void set_element(T _val, VNT _pos);

    void set_all_constant(T _val);

    void fill_with_zeros() { nvals = 0; };

    void convert(DenseVector<T> *_dense_vector);

    VNT get_size() const {return size;};
private:
    VNT size;
    VNT nvals;

    VNT *ids;
    T *vals;
};

#include "sparse_vector.hpp"

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
