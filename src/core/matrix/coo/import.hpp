/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCOO<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _non_zeroes_num)
{
    resize(_size, _non_zeroes_num);

    size = _size;
    non_zeroes_num = _non_zeroes_num;
    MemoryAPI::copy(row_ids, _row_ids, _non_zeroes_num);
    MemoryAPI::copy(col_ids, _col_ids, _non_zeroes_num);
    MemoryAPI::copy(vals, _vals, _non_zeroes_num);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
