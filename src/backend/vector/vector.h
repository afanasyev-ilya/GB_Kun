#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "generic_vector.h"
#include "dense_vector/dense_vector.h"
#include "sparse_vector/sparse_vector.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template<typename T>
class Vector {
public:
    Vector(VNT _size)
    {
        storage = GrB_SPARSE;
        main_container = new SparseVector<T>(size);
        secondary_container = new DenseVector<T>(size);
    }

    ~Vector()
    {
        delete main_container;
        delete secondary_container;
    }

    void set_constant(T _val)
    {
        if(_val == 0) // if val is zero, it is sparse vector with 0 elements
        {
            storage = GrB_SPARSE;
            main_container->fill_with_zeros();
        }
        else
        {
            if(is_dense()) // if dense, just change all contents
            {
                main_container->set_all_constant(_val);
            }
            else // convert to dense
            {
                storage = GrB_DENSE;
                swap(main_container, secondary_container);
                main_container->set_all_constant(_val);
            }
        }
    };

    DenseVector<T>* getDense()
    {
        if(is_dense())
            return (DenseVector<T>*)main_container;
        else
        {
            cout << "conversion required" << endl;
            throw "Error in getDense, conversion not implemented";
        }
    }

    SparseVector<T>* getSparse()
    {
        if(is_sparse())
            return (SparseVector<T>*)main_container;
        else
        {
            cout << "conversion required" << endl;
            throw "Error in getDense, conversion not implemented";
        }
    }

    const DenseVector<T>* getDense() const
    {
        if(is_dense())
            return (DenseVector<T>*)main_container;
        else
        {
            cout << "conversion required" << endl;
            throw "Error in getDense, conversion not implemented";
        }
    }

    const SparseVector<T>* getSparse() const
    {
        if(is_sparse())
            return (SparseVector<T>*)main_container;
        else
        {
            cout << "conversion required" << endl;
            throw "Error in getDense, conversion not implemented";
        }
    }

    void getStorage(Storage* _storage) const
    {
        *_storage = storage;
    }

    void setStorage(Storage _storage)
    {
        storage = _storage;
    }

    void set_element(T _val, VNT _pos)
    {
        main_container->set_element(_val, _pos);
    }

    bool is_sparse() const { return storage == GrB_SPARSE;};
    bool is_dense() const { return storage == GrB_DENSE;};

    LA_Info build (const Index* _indices,
                   const T*     _values,
                   Index _nvals)
    {
        storage = GrB_SPARSE;
        auto *sparse_vec = (SparseVector<T>*)main_container;
        return sparse_vec->build(_indices, _values, _nvals);
    }

    LA_Info build(const T*    _values,
                  Index _nvals)
    {
        storage = GrB_DENSE;
        auto *dense_vec = (DenseVector<T>*)main_container;
        return dense_vec->build(_values, _nvals);
    }

    void print() const
    {
        main_container->print();
    }

    void print_storage_type() const
    {
        main_container->print_storage_type();
    }

    VNT get_nvals() const
    {
        return main_container->get_nvals();
    };
private:
    VNT size;
    VNT nnz;
    Storage storage;

    GenericVector<T> *main_container;
    GenericVector<T> *secondary_container;

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    if(lhs.storage != rhs.storage) // storages mismatch, not equal
    {
        return 0;
    }
    else
    {
        if(lhs.is_dense())
        {
            auto den_lhs = (DenseVector<T> *)lhs.main_container;
            auto den_rhs = (DenseVector<T> *)lhs.main_container;
            return (*den_lhs) == (*den_rhs);
        }
        else
        {
            throw " == for sparse vectors not implemented yet";
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


