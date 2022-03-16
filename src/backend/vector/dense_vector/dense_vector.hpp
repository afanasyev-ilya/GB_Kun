#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DenseVector<T>::DenseVector(VNT _size)
{
    size = _size;
    MemoryAPI::allocate_array(&vals, size);
    #pragma omp parallel for schedule(static) // numa aware alloc
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
    vals = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::print() const
{
    for(VNT i = 0; i < size; i++)
    {
        if(fabs(vals[i]) >= 1000000/*std::numeric_limits<T>::max()*/)
            cout << "[" << i << "]:" << "inf" << " ";
        else
            cout << "[" << i << "]:" << vals[i] << " ";
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
VNT DenseVector<T>::get_nvals() const
{
    VNT loc_nvals = 0;
    #pragma omp parallel for reduction(+: loc_nvals)
    for(int i = 0; i < get_size(); i++)
        if(vals[i] != 0)
            loc_nvals++;
    return loc_nvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::set_element(T _val, VNT _pos)
{
    vals[_pos] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::set_all_constant(T _val)
{
    #pragma omp parallel for schedule(static)
    for (VNT i = 0; i < size; i++)
    {
        vals[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::fill_with_zeros()
{
    #pragma omp parallel for schedule(static)
    for (VNT i = 0; i < size; i++) {
        vals[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DenseVector<T>::convert(SparseVector<T> *_sparse_vector)
{
    #ifdef __DEBUG_INFO__
    cout << "converting sparse -> dense, name = " << this->name << endl;
    #endif
    memset(this->vals, 0, size*sizeof(T));

    VNT *sparse_ids = _sparse_vector->get_ids();
    T *sparse_vals = _sparse_vector->get_vals();
    VNT sparse_nvals = _sparse_vector->get_nvals();

    #pragma omp parallel for
    for(VNT i = 0; i < sparse_nvals; i++)
        this->vals[sparse_ids[i]] = sparse_vals[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
