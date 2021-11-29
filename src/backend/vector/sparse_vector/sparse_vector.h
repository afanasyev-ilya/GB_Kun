#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SparseVector
{
public:
    SparseVector(int _size);

    ~SparseVector();

    void set_constant(T _val);

    void print();
private:
    VNT size;
    VNT nz;

    T *vals;
    VNT *ids;

    template<typename Y>
    friend bool operator==(SparseVector<Y>& lhs, SparseVector<Y>& rhs);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
SparseVector<T>::SparseVector(int _size)
{
    size = _size;
    MemoryAPI::allocate_array(&vals, size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
SparseVector<T>::~SparseVector()
{
    MemoryAPI::free_array(vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::set_constant(T _val)
{
    for(VNT i = 0; i < size; i++)
    {
        if (i % 10 == 0) {
            int pos = i / 10;
            vals[pos] = _val;
            ids[pos] = i;
            nz++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::print()
{
    for(VNT i = 0; i < size; i++)
    {
        cout << vals[i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool operator==(SparseVector<T>& lhs, SparseVector<T>& rhs)
{
    if(lhs.size != rhs.size)
        return false;

    VNT error_count = 0;
    for(VNT i = 0; i < lhs.size; i++)
    {
        if(fabs(lhs.vals[i] - rhs.vals[i]) > 0.0001 && lhs.vals[i] < 100000)
        {
            if(error_count < 20)
                cout << "Error in " << i << " : " << lhs.vals[i] << " " << rhs.vals[i] << endl;
            error_count++;
        }
    }

    for(VNT i = 0; i < min(lhs.size, CHECK_PRINT_NUM); i++)
    {
        cout << "check " << i << " : " << lhs.vals[i] << " vs " << rhs.vals[i] << endl;
    }

    cout << "error_count: " << error_count << "/" << max(lhs.size, rhs.size)  << endl;
    if(error_count == 0)
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
