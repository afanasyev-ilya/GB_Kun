#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DenseVector
{
private:
    VNT size;
    T *vals;
public:
    DenseVector(int _size)
    {
        size = _size;
        MemoryAPI::allocate_array(&vals, size);
    }

    ~DenseVector()
    {
        MemoryAPI::free_array(vals);
    }

    T *get_vals() {return vals;};
    VNT get_size() {return size;};

    void set_constant(T _val)
    {
        for(VNT i = 0; i < size; i++)
        {
            vals[i] = _val;
        }
    }

    void print()
    {
        for(VNT i = 0; i < size; i++)
        {
            cout << vals[i] << " ";
        }
        cout << endl;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(const DenseVector<T>& lhs, const DenseVector<T>& rhs)
{
    if(lhs.get_size() != rhs.get_size())
        return false;

    for(VNT i = 0; i < lhs.get_size(); i++)
    {
        if(fabs(lhs.get_vals()[i] - rhs.get_vals()[i]) > 0.0001)
        {
            cout << "Error in " << i << " : " << lhs.get_vals()[i] << " " << rhs.get_vals()[i] << endl;
            return false;
        }
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
