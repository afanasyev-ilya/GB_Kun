#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SPARSITY_K 100

template <typename T>
class SparseVector
{
public:
    SparseVector(int _size)
    {
        size = _size;
        nnz = _size/SPARSITY_K;
        //MemoryAPI::allocate_array(&vals, nnz);
        //MemoryAPI::allocate_array(&ids, nnz);
    };

    ~SparseVector()
    {
        //MemoryAPI::free_array(vals);
        //MemoryAPI::free_array(ids);
    };

    void set_constant(T _val)
    {
        for(VNT i = 0; i < size; i++)
        {
            if (i % SPARSITY_K == 0) {
                int pos = i / SPARSITY_K;
                vals[pos] = _val;
                ids[pos] = i;
                nnz++;
            }
        }
    };

    void print() const {
        for(VNT i = 0; i < nnz; i++)
        {
            cout << "( "<< ids[i]<< " , "  << vals[i] << ") ";
        }
        cout << endl;

    };

    void get_nnz(VNT* _nnz) const {
        *_nnz = nnz;
    }

    void get_size(VNT* _size) const {
        *_size = size;
    }

    void set_nnz(VNT _nnz) const {
        nnz = _nnz;
    }

    void set_size(VNT _size) const {
        size = _size;
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

    void set_ids(VNT *_ids) const {
       ids = _ids;
    }

    void set_vals(T* _vals) const {
        vals = _vals;
    }

    /* TODO implement constructor and build for sparse */

    LA_Info build(const Index* indices, const T *values, Index nvals) {

        if (nvals > size)
            return GrB_INDEX_OUT_OF_BOUNDS;

        for (Index i = 0; i < nvals; i++){
            vals[i] = values[i];
            ids[i] = indices[i];
        }
        return GrB_SUCCESS;
    }

    void swap(SparseVector* rhs) const {
        VNT tmp_size = size;
        VNT tmp_nnz = nnz;
        T* tmp_vals = vals;
        VNT* tmp_ids = ids;

        rhs->get_size(&size);
        rhs->get_nnz(&nnz);
        vals = rhs->get_vals();
        ids = rhs->get_ids();

        rhs->set_nnz(tmp_nnz);
        rhs->set_size(tmp_size);
        rhs->set_ids(tmp_ids);
        rhs->set_vals(tmp_vals);
    }

private:
    mutable VNT size;
    mutable ENT nnz;

    mutable T *vals;
    mutable VNT *ids;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
