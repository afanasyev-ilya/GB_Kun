#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCOO<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    resize(_size, _nnz);
    size = _size;
    nnz = _nnz;

    VNT* col_ids_new = *(&col_ids);
    VNT* row_ids_new = *(&row_ids);
    T* vals_new = *(&vals);

    MemoryAPI::copy(row_ids_new, _row_ids, _nnz);
    MemoryAPI::copy(col_ids_new, _col_ids, _nnz);
    MemoryAPI::copy(vals_new, _vals, _nnz);

    bool use_cache_blocking = true;
    if(use_cache_blocking)
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nnz);

        #pragma omp parallel for
        for(ENT i = 0; i < _nnz; i++)
            sort_indexes[i] = i;

        int seg_size = 64*1024 / sizeof(T);
        cout << "num segments: " << (size - 1)/seg_size + 1 << endl;

        std::sort(sort_indexes, sort_indexes + _nnz,
                  [_row_ids, _col_ids, seg_size](int index1, int index2)
                  {
                          if(_row_ids[index1] / seg_size == _row_ids[index2] / seg_size)
                              return _col_ids[index1] / seg_size < _col_ids[index2] / seg_size;
                          else
                              return _row_ids[index1] / seg_size < _row_ids[index2] / seg_size;
                  });
        reorder(row_ids_new, sort_indexes, _nnz);
        reorder(col_ids_new, sort_indexes, _nnz);
        reorder(vals_new, sort_indexes, _nnz);

        MemoryAPI::free_array(sort_indexes);
    }

    /*for(ENT i = 0; i < nnz; i++)
    {

    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
