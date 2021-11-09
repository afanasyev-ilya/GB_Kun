/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void reorder(T *data, ENT *indexes, ENT size)
{
    T *tmp;
    MemoryAPI::allocate_array(&tmp, size);

    for(ENT i = 0; i < size; i++)
        tmp[i] = data[indexes[i]];

    for(ENT i = 0; i < size; i++)
        data[i] = tmp[i];

    MemoryAPI::free_array(tmp);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCOO<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz, bool _optimized)
{
    resize(_size, _nz);
    size = _size;
    nz = _nz;

    if(_optimized)
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nz);
        for(ENT i = 0; i < _nz; i++)
            sort_indexes[i] = i;

        int seg_size = 1024*1024 / sizeof(int);

        cout << "num segments: " << (size - 1)/seg_size + 1 << endl;

        std::sort(sort_indexes, sort_indexes + _nz,
                  [_row_ids, _col_ids, seg_size](int index1, int index2)
                  {
                          if(_row_ids[index1] / seg_size == _row_ids[index2] / seg_size)
                              return _col_ids[index1] / seg_size < _col_ids[index2] / seg_size;
                          else
                              return _row_ids[index1] / seg_size < _row_ids[index2] / seg_size;
                  });

        reorder(_row_ids, sort_indexes, _nz);
        reorder(_col_ids, sort_indexes, _nz);
        reorder(_vals, sort_indexes, _nz);

        MemoryAPI::free_array(sort_indexes);
    }

    MemoryAPI::copy(row_ids, _row_ids, _nz);
    MemoryAPI::copy(col_ids, _col_ids, _nz);
    MemoryAPI::copy(vals, _vals, _nz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
