/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::MatrixCSR()
{
    target_socket = 0;
    alloc(1, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::~MatrixCSR()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::alloc(VNT _size, ENT _nnz)
{
    this->size = _size;
    this->nnz = _nnz;

    MemoryAPI::allocate_array(&row_ptr, this->size + 1);
    MemoryAPI::allocate_array(&col_ids, this->nnz);
    MemoryAPI::allocate_array(&vals, this->nnz);
    MemoryAPI::allocate_array(&tmp_buffer, this->size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::free()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
    MemoryAPI::free_array(tmp_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::resize(VNT _size, ENT _nnz)
{
    this->free();
    this->alloc(_size, _nnz);
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
    cout << "--------------------\n";
    cout << "MATRIX: [ " << endl;
    for(VNT row = 0; row < size; row++)
    {
        for(VNT col = 0; col < size; col++)
        {
            cout << get(row, col) << " ";
        }
        cout << endl;
    }
    cout << "]\n";
    cout << "--------------------\n";
    cout << "nnz: " << nnz << endl;
    cout << "col_ids: [ ";
    for (int i = 0; i < get_nnz(); i++) {
        cout << get_col_ids()[i] << " ";
    }
    cout << "]\n";
    cout << endl;
    Index size_ = 0;
    get_size(&size_);
    cout << "row_ptr: [ ";
    for (int i = 0; i < size_ + 1; i++) {
        cout << get_row_ptr()[i] << " ";
    }
    cout << "]" << endl;
    cout << "--------------------\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
