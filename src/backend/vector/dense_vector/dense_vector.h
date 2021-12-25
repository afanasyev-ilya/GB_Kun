#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{


template <typename T>
class DenseVector
{
public:
    DenseVector(VNT _size);

    ~DenseVector();

    void set_constant(T _val);

    void print() const;

    void get_size (VNT* _size) const {
        *_size = size;
    }

    VNT get_size () const {
        return size;
    }

    T* get_vals () {
        return vals;
    }

    const T* get_vals () const {
        return vals;
    }

    void set_size(VNT _size) const {
        size = _size;
    }

    void set_vals(T* _vals) const {
        vals = _vals;
    }

    LA_Info build(const T* values,
                  VNT nvals) {
        if (nvals > size){
            return GrB_INDEX_OUT_OF_BOUNDS;
        }
        #pragma omp parallel for schedule(static)
        for (VNT i = 0; i < nvals; i++) {
            vals[i] = values[i];
        }
        return GrB_SUCCESS;
    }

private:
    mutable VNT size;
    mutable T *vals;

    template<typename Y>
    friend bool operator==(DenseVector<Y>& lhs, DenseVector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DenseVector<T>::DenseVector(VNT _size)
{
    size = _size;
    MemoryAPI::allocate_array(&vals, size);
    #pragma omp parallel for schedule(static)
    for (VNT i = 0; i < size; i++)
    {
        vals[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DenseVector<T>::~DenseVector()
{
    MemoryAPI::free_array(vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::set_constant(T _val)
{
    for(VNT i = 0; i < size; i++)
    {
        vals[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::print() const
{
    for(VNT i = 0; i < size; i++)
    {
        cout << vals[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(DenseVector<T>& lhs, DenseVector<T>& rhs)
{
    if(lhs.size != rhs.size)
        return false;

    VNT error_count = 0;
    for(VNT i = 0; i < lhs.size; i++)
    {
        if(fabs(lhs.vals[i] - rhs.vals[i]) > 0.001 && lhs.vals[i] < 100000)
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
    if(error_count == 0)
        return true;
    else
        return false;
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
