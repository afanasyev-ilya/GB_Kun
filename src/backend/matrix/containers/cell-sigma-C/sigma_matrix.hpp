/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
MatrixCellSigmaC<T>::MatrixCellSigmaC()
{
    alloc(1, 1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
MatrixCellSigmaC<T>::~MatrixCellSigmaC()
{
    free();
    delete []vertex_groups;
    delete []cell_c_vertex_groups;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCellSigmaC<T>::alloc(VNT _size, ENT _nz)
{
    size = _size;
    nz = _nz;

    MemoryAPI::allocate_array(&row_ptr, size + 1);
    MemoryAPI::allocate_array(&col_ids, nz);
    MemoryAPI::allocate_array(&vals, nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCellSigmaC<T>::free()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCellSigmaC<T>::resize(VNT _size, ENT _nz)
{
    this->free();
    this->alloc(_size, _nz);
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