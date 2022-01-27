#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "generic_vector.h"
#include "dense_vector/dense_vector.h"
#include "sparse_vector/sparse_vector.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T>
void ptr_swap(T *& _a, T *& _b)
{
    T* c = _a;
    _a = _b;
    _b = c;
}

template<typename T>
class Vector {
public:
    Vector(VNT _size)
    {
        storage = GrB_SPARSE;
        main_container = new SparseVector<T>(_size);
        secondary_container = new DenseVector<T>(_size);
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
                swap_to_dense();
                main_container->set_all_constant(_val);
            }
        }
    };

    DenseVector<T>* getDense()
    {
        (const_cast <Vector<T>*> (this))->force_to_dense();
        return (DenseVector<T>*)main_container;
    }

    SparseVector<T>* getSparse()
    {
        (const_cast <Vector<T>*> (this))->force_to_sparse();
        return (SparseVector<T>*)main_container;
    }

    const DenseVector<T>* getDense() const
    {
        (const_cast <Vector<T>*> (this))->force_to_dense();
        return (DenseVector<T>*)main_container;
    }

    const SparseVector<T>* getSparse() const
    {
        (const_cast <Vector<T>*> (this))->force_to_sparse();
        return (SparseVector<T>*)main_container;
    }

    void force_to_dense()
    {
        if(is_dense())
            return;
        else
        {
            swap_to_dense();
            ((DenseVector<T>*)main_container)->convert((SparseVector<T>*)secondary_container);
        }
    }

    void force_to_sparse()
    {
        if(is_sparse())
            return;
        else
        {
            swap_to_sparse();
            ((SparseVector<T>*)main_container)->convert((DenseVector<T>*)secondary_container);
        }
    }

    void convert_if_required()
    {
        VNT nvals = main_container->get_nvals();
        if(nvals > main_container->get_size() * SPARSE_VECTOR_THRESHOLD) // TODO more complex
        {
            force_to_dense();
        }
        else
        {
            force_to_sparse();
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
        swap_to_sparse();
        ((SparseVector<T>*)main_container)->build(_indices, _values, _nvals);

        SparseVector<T>* sparse_vec = ((SparseVector<T>*)main_container);

        sparse_vec->fill_with_zeros();
        sparse_vec->build(_indices, _values, _nvals);
        return GrB_SUCCESS;
    }

    LA_Info build(const T*    _values,
                  Index _nvals)
    {
        swap_to_dense();
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

    VNT get_size() const
    {
        return main_container->get_size();
    }

    void swap(Vector *_another)
    {
        ptr_swap(this->main_container, _another->main_container);
        ptr_swap(this->secondary_container, _another->secondary_container);
        std::swap(this->nnz, _another->nnz);
        std::swap(this->storage, _another->storage);
    }
private:
    VNT nnz;
    Storage storage;

    GenericVector<T> *main_container;
    GenericVector<T> *secondary_container;

    void swap_to_sparse()
    {
        if(is_dense())
        {
            ptr_swap(main_container, secondary_container);
            storage = GrB_SPARSE;
        }
    }

    void swap_to_dense()
    {
        if(is_sparse())
        {
            ptr_swap(main_container, secondary_container);
            storage = GrB_DENSE;
        }
    }

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    if(lhs.storage != rhs.storage) // storages mismatch, not equal
    {
        cout << "Different storage!\n";
        return 0;
    }
    else
    {
        if(lhs.is_dense())
        {
            auto den_lhs = (DenseVector<T> *)lhs.main_container;
            auto den_rhs = (DenseVector<T> *)rhs.main_container;
            return (*den_lhs) == (*den_rhs);
        }
        else
        {
            throw " == for sparse vectors not implemented yet";
        }
    }
}

template <typename T>
bool operator!=(Vector<T>& lhs, Vector<T>& rhs)
{
    return !(lhs == rhs);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


