/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::MatrixCSR()
{
    target_socket = 0;
    alloc(1, 1, target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::~MatrixCSR()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::alloc(VNT _size, ENT _nnz, int _target_socket)
{
    this->size = _size;
    this->nnz = _nnz;
    target_socket = _target_socket;

    MemoryAPI::numa_aware_alloc(&row_ptr, this->size + 1, _target_socket);
    MemoryAPI::numa_aware_alloc(&col_ids, this->nnz, _target_socket);
    MemoryAPI::numa_aware_alloc(&vals, this->nnz, _target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::free()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::resize(VNT _size, ENT _nnz, int _target_socket)
{
    this->free();
    this->alloc(_size, _nnz, _target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCSR<T>::is_non_zero(VNT _row, VNT _col)
{
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(col_ids[i] == _col)
            return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T MatrixCSR<T>::get(VNT _row, VNT _col) const
{
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(col_ids[i] == _col)
            return vals[i];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::print() const
{
    for(VNT row = 0; row < size; row++)
    {
        for(VNT col = 0; col < size; col++)
        {
            cout << get(row, col) << " ";
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
