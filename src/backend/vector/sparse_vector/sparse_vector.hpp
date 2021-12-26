#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::set_element(T _val, VNT _pos)
{
    for(VNT i = 0; i < nnz; i++)
    {
        if(ids[i] == _pos)
        {
            vals[i] = _val;
            return;
        }
    }
    ids[nnz] = _pos;
    vals[nnz] = _val;
    nnz++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::set_all_constant(T _val)
{
    #pragma omp parallel for
    for(VNT i = 0; i < nnz; i++)
    {
        vals[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::convert(DenseVector<T> *_dense_vector)
{
    cout << "converting dense -> sparse" << endl;
    VNT dense_size = _dense_vector->get_size();
    T* dense_vals = _dense_vector->get_vals();

    // TODO faster using optimized copy if
    nnz = 0;
    for(VNT i = 0; i < size; i++)
    {
        if(dense_vals[i] != 0)
        {
            vals[nnz] = dense_vals[i];
            ids[nnz] = i;
            nnz++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

