#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::set_element(T _val, VNT _pos)
{
    for(VNT i = 0; i < nvals; i++)
    {
        if(ids[i] == _pos)
        {
            vals[i] = _val;
            return;
        }
    }
    ids[nvals] = _pos;
    vals[nvals] = _val;
    nvals++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::set_all_constant(T _val)
{
    #pragma omp parallel for
    for(VNT i = 0; i < nvals; i++)
    {
        vals[i] = _val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SparseVector<T>::convert(DenseVector<T> *_dense_vector)
{
    LOG_TRACE("converting dense -> sparse, name = " + this->name)
    VNT dense_size = _dense_vector->get_size();
    T* dense_vals = _dense_vector->get_vals();

    nvals = 0;
    const int max_threads = 64*2;
    VNT sum_array[max_threads + 1];
    if(omp_get_max_threads() > max_threads)
        throw " Error in SparseVector<T>::convert : max_threads = 128 is too small for this architecture, please increase";
    sum_array[0] = 0;
    #pragma omp parallel
    {
        VNT tid_elements = 0;
        #pragma omp for nowait
        for(VNT i = 0; i < size; i++)
        {
            if(dense_vals[i] != 0)
            {
                tid_elements++;
            }
        }

        const int tid = omp_get_thread_num();

        sum_array[tid + 1] = tid_elements;

        #pragma omp barrier

        VNT offset = 0;
        for(VNT i = 0; i < (tid + 1); i++)
        {
            offset += sum_array[i];
        }

        tid_elements = 0;
        Index *ids_ptr = &ids[offset];
        T *vals_ptr = &vals[offset];

        #pragma omp for nowait
        for(VNT i = 0; i < size; i++)
        {
            if(dense_vals[i] != 0)
            {
                ids_ptr[tid_elements] = i;
                vals_ptr[tid_elements] = dense_vals[i];
                tid_elements++;
            }
        }

        #pragma omp atomic
        nvals += tid_elements;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

