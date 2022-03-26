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
        //cout << nvals << " vs " << (VNT)(get_size() * SPARSE_VECTOR_THRESHOLD) << " at vector " << get_name() << endl;
        if(nvals > (VNT)(get_size() * SPARSE_VECTOR_THRESHOLD)) // TODO more complex
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
        *_storage = main_container->get_storage();
    }

    Storage get_storage() const
    {
        return main_container->get_storage();
    }

    void set_element(T _val, VNT _pos)
    {
        main_container->set_element(_val, _pos);
    }

    void set_name(const string &_name) { main_container->set_name(_name); secondary_container->set_name(_name); };
    const string &get_name() const { return main_container->get_name(); };

    bool is_sparse() const { return get_storage() == GrB_SPARSE;};
    bool is_dense() const { return get_storage() == GrB_DENSE;};

    void print_threshold_info() const
    {
        VNT nvals = main_container->get_nvals();

        if(nvals > (VNT)(get_size() * SPARSE_VECTOR_THRESHOLD))
        {
            cout << "vector " << get_name() << " is DENSE since it contains > " << (100.0*nvals)/get_size() << " > " << 100.0*SPARSE_VECTOR_THRESHOLD << " % elems" << endl;
        }
        else
        {
            cout << "vector " << get_name() << " is SPARSE since it contains " << (100.0*nvals)/get_size() << " <= " << 100.0*SPARSE_VECTOR_THRESHOLD << " % elems" << endl;
        }
    }

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
        #ifndef __SHORT_VECTOR_PRINT__
        if(is_dense())
            cout << "vector is dense" << endl;
        else
            cout << "vector is sparse" << endl;
        cout << "nvals: " << main_container->get_nvals() << " / " << main_container->get_size() << endl;
        #endif
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

    LA_Info fillAscending(Index nvals) {
        force_to_dense();
        return main_container->fillAscending(nvals);
    }

    LA_Info dup(const Vector<T>* rhs) {
        if(rhs->is_dense())
            this->swap_to_dense();
        else
            this->swap_to_sparse();
        main_container->dup(rhs->main_container);
        return GrB_SUCCESS;
    }

    void swap(Vector *_another)
    {
        ptr_swap(this->main_container, _another->main_container);
        ptr_swap(this->secondary_container, _another->secondary_container);
    }

    LA_Info clear()
    {
        swap_to_sparse();
        main_container->fill_with_zeros();
        return GrB_SUCCESS;
    }

    T const & get_at(Index _index) const
    {
        DenseVector<T> *dense_data = (const_cast <Vector<T>*> (this))->getDense();
        const T* vals = dense_data->get_vals();
        if(_index >= 0 && _index < dense_data->get_size())
            return vals[_index];
        else
            throw "Error: out of range in backend::vector";
    }
private:
    GenericVector<T> *main_container;
    GenericVector<T> *secondary_container;

    void swap_to_sparse()
    {
        if(is_dense())
        {
            ptr_swap(main_container, secondary_container);
        }
    }

    void swap_to_dense()
    {
        if(is_sparse())
        {
            ptr_swap(main_container, secondary_container);
        }
    }

    template<typename Y>
    friend bool operator==(Vector<Y>& lhs, Vector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(Vector<T>& lhs, Vector<T>& rhs)
{
    if(lhs.is_sparse())
        lhs.force_to_dense();
    if(rhs.is_sparse())
        rhs.force_to_dense();

    auto den_lhs = (DenseVector<T> *)lhs.main_container;
    auto den_rhs = (DenseVector<T> *)rhs.main_container;
    return (*den_lhs) == (*den_rhs);
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


