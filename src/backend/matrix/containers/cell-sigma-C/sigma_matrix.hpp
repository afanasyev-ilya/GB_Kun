/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCellSigmaC<T>::MatrixCellSigmaC()
{
    target_socket = 0;
    alloc(1, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCellSigmaC<T>::~MatrixCellSigmaC()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::alloc(VNT _size, ENT _nz)
{
    this->size = _size;
    this->nz = _nz;

    MemoryAPI::allocate_array(&row_ptr, this->size + 1);
    MemoryAPI::allocate_array(&col_ids, this->nz);
    MemoryAPI::allocate_array(&vals, this->nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::free()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCellSigmaC<T>::resize(VNT _size, ENT _nz)
{
    this->free();
    this->alloc(_size, _nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCellSigmaC<T>::is_non_zero(VNT _row, VNT _col)
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
T MatrixCellSigmaC<T>::get(VNT _row, VNT _col)
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
void MatrixCellSigmaC<T>::print()
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
