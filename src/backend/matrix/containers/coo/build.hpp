/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void reorder(T *data, ENT *indexes, ENT size)
{
    T *tmp;
    MemoryAPI::allocate_array(&tmp, size);

    #pragma omp parallel for
    for(ENT i = 0; i < size; i++)
        tmp[i] = data[indexes[i]];

    #pragma omp parallel for
    for(ENT i = 0; i < size; i++)
        data[i] = tmp[i];

    MemoryAPI::free_array(tmp);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCOO<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    resize(_size, _nz);
    size = _size;
    nz = _nz;

    VNT* col_ids_new = *(&row_ids);
    VNT* row_ids_new = *(&col_ids);
    T* vals_new = *(&vals);

    bool save_to_file = false;
    if(save_to_file)
    {
        MemoryAPI::copy(row_ids_new, _row_ids, _nz);
        MemoryAPI::copy(col_ids_new, _col_ids, _nz);
        MemoryAPI::copy(vals_new, _vals, _nz);

        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nz);

        #pragma omp parallel for
        for(ENT i = 0; i < _nz; i++)
            sort_indexes[i] = i;

        std::sort(sort_indexes, sort_indexes + _nz,
                  [_row_ids, _col_ids](int index1, int index2)
                  {
                      if(_row_ids[index1] == _row_ids[index2])
                          return _col_ids[index1] < _col_ids[index2];
                      else
                          return _row_ids[index1] < _row_ids[index2];
                  });
        reorder(row_ids_new, sort_indexes, _nz);
        reorder(col_ids_new, sort_indexes, _nz);
        reorder(vals_new, sort_indexes, _nz);

        MemoryAPI::free_array(sort_indexes);

        ENT unique_edges = 0;
        for(ENT i = 1; i < _nz; i++)
            if((row_ids[i] != row_ids[i - 1]) && (col_ids[i] != col_ids[i - 1]) && (col_ids[i] != row_ids[i]))
                unique_edges++;

        ofstream matrix_file;
        matrix_file.open ("last_synth_graph.mtx");
        matrix_file << "%%MatrixMarket matrix coordinate pattern general" << endl;
        matrix_file << size << " " << size << " " << unique_edges << endl;
        for(ENT i = 1; i < _nz; i++)
        {
            if((row_ids[i] != row_ids[i - 1]) && (col_ids[i] != col_ids[i - 1]) && (col_ids[i] != row_ids[i]))
                matrix_file << row_ids[i] + 1 << " " << col_ids[i] + 1 << endl;
        }
        matrix_file.close();
    }

    MemoryAPI::copy(row_ids_new, _row_ids, _nz);
    MemoryAPI::copy(col_ids_new, _col_ids, _nz);
    MemoryAPI::copy(vals_new, _vals, _nz);

    bool _optimized = false;
    if(_optimized)
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nz);

        #pragma omp parallel for
        for(ENT i = 0; i < _nz; i++)
            sort_indexes[i] = i;

        int seg_size = 512*1024 / sizeof(float);
        cout << "num segments: " << (size - 1)/seg_size + 1 << endl;

        std::sort(sort_indexes, sort_indexes + _nz,
                  [_row_ids, _col_ids, seg_size](int index1, int index2)
                  {
                          if(_row_ids[index1] / seg_size == _row_ids[index2] / seg_size)
                              return _col_ids[index1] / seg_size < _col_ids[index2] / seg_size;
                          else
                              return _row_ids[index1] / seg_size < _row_ids[index2] / seg_size;
                  });
        reorder(row_ids_new, sort_indexes, _nz);
        reorder(col_ids_new, sort_indexes, _nz);
        reorder(vals_new, sort_indexes, _nz);

        MemoryAPI::free_array(sort_indexes);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
