#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DenseVector
{
public:
    DenseVector(int _size);

    ~DenseVector();

    T *get_vals() {return vals;};
    VNT get_size() {return size;};

    void set_constant(T _val);

    void print();

    template<typename Y>
    friend void SpMV(MatrixCSR<Y> &_matrix,
                     DenseVector<Y> &_x,
                     DenseVector<Y> &_y);
private:
    VNT size;
    T *vals;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DenseVector<T>::DenseVector(int _size)
{
    size = _size;
    MemoryAPI::allocate_array(&vals, size);
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
void DenseVector<T>::print()
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
    if(lhs.get_size() != rhs.get_size())
        return false;

    VNT error_count = 0;
    for(VNT i = 0; i < lhs.get_size(); i++)
    {
        if(fabs(lhs.get_vals()[i] - rhs.get_vals()[i]) > 0.0001)
        {
            if(error_count < 20)
                cout << "Error in " << i << " : " << lhs.get_vals()[i] << " " << rhs.get_vals()[i] << endl;
            error_count++;
        }
    }

    cout << "error_count: " << error_count << "/" << max(lhs.get_size(), rhs.get_size())  << endl;
    if(error_count == 0)
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
