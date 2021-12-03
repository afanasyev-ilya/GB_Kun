#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SparseVector
{
public:
    SparseVector(int _size) {};

    ~SparseVector() {};

    void set_constant(T _val) {};

    void print() {
        for(VNT i = 0; i < nz; i++)
        {
            cout << "( "<< ids[i]<< " , "  << vals[i] << ") ";
        }
        cout << endl;

    };

    void get_nz(VNT* _nz) const {
        *_nz = nz;
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

private:
    VNT size;
    VNT nz;

    T *vals;
    VNT *ids;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
