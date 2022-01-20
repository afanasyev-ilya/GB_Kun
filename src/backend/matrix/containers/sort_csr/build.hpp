/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSortCSR<T>::build(VNT *_row_degrees,
                             VNT *_col_degrees,
                             VNT _nrows,
                             VNT _ncols,
                             ENT _nnz,
                             const ENT *_row_ptr,
                             const VNT *_col_ids,
                             const T *_vals,
                             int _target_socket)
{
    double t1, t2;
    t1 = omp_get_wtime();

    size = _nrows;
    nnz = _nnz;
    target_socket = _target_socket;

    VNT *col_conversion_indexes, *row_conversion_indexes;
    MemoryAPI::allocate_array(&col_conversion_indexes, size);
    MemoryAPI::allocate_array(&row_conversion_indexes, size);
    MemoryAPI::allocate_array(&col_backward_conversion, size);

    for(VNT i = 0; i < size; i++)
    {
        col_conversion_indexes[i] = i;
        row_conversion_indexes[i] = i;
    }

    std::sort(col_conversion_indexes, col_conversion_indexes + size,
              [_col_degrees](int index1, int index2)
              {
                  return _col_degrees[index1] > _col_degrees[index2];
              });

    std::sort(row_conversion_indexes, row_conversion_indexes + size,
              [_row_degrees](int index1, int index2)
              {
                  return _row_degrees[index1] > _row_degrees[index2];
              });

    for(VNT i = 0; i < size; i++)
    {
        col_backward_conversion[col_conversion_indexes[i]] = i;
    }
    t2 = omp_get_wtime();
    cout << "sorting and conversion array init time: " << t2 - t1 << " sec" << endl;

    resize(size, nnz, target_socket);

    t1 = omp_get_wtime();
    row_ptr[0] = 0;
    #pragma omp parallel for schedule(static)
    for(VNT row = 0; row < size; row++)
    {
        VNT old_row = row_conversion_indexes[row];
        VNT connections_count = _row_ptr[old_row + 1] - _row_ptr[old_row];
        row_ptr[row + 1] = connections_count;
    }

    for(VNT row = 1; row < size + 1; row++)
    {
        row_ptr[row] += row_ptr[row - 1];
    }

    #pragma omp parallel for schedule(guided, 1024)
    for(VNT row = 0; row < size; row++)
    {
        VNT old_row = row_conversion_indexes[row];
        VNT connections_count = _row_ptr[old_row + 1] - _row_ptr[old_row];

        for(ENT j = 0; j < connections_count; j++)
        {
            VNT old_col_id = _col_ids[_row_ptr[old_row] + j];
            T old_val = _vals[_row_ptr[old_row] + j];

            VNT new_col_id = col_backward_conversion[old_col_id];
            col_ids[row_ptr[row] + j] = new_col_id;
            vals[row_ptr[row] + j] = old_val;
        }
    }
    t2 = omp_get_wtime();
    cout << "sorted CSR generation time: " << t2 - t1 << " sec" << endl;

    /*int part_size = 2;

    while((part_size <= 8192) && (part_size < _size))
    {
        ENT cnt = 0;
        for(VNT x = 0; x < part_size; x++)
        {
            for(VNT y = 0; y < part_size; y++)
            {
                T val = get(x, y);
                if(val != 0)
                    cnt++;
            }
        }
        cout << cnt << " " << part_size*part_size << " | " << _nnz << endl;
        cout << "dense percent: " << (100.0*cnt) / (part_size * part_size) << " % for part = " << part_size << endl;
        cout << "perc of all: " << (100.0*cnt) / _nnz << " % of all edges" << endl;
        cout << sizeof(T)*part_size*part_size/1e9 << " GB is size of dense part" << endl << endl;
        part_size *= 2;
    }*/

    MemoryAPI::free_array(col_conversion_indexes);
    MemoryAPI::free_array(row_conversion_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
