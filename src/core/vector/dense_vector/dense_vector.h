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
