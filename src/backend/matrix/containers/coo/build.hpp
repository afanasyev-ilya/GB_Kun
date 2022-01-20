#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline VNT row_block(VNT _id)
{
    int seg_size = 16*1024 / sizeof(T);
    return _id / seg_size;
}

template <typename T>
inline VNT col_block(VNT _id)
{
    int seg_size = 512*1024 / sizeof(T);
    return _id / seg_size;
}

template <typename T>
void MatrixCOO<T>::build(VNT _num_rows, ENT _nnz, const ENT *_row_ptr, const VNT *_col_ids, const T *_vals, int _socket)
{
    int num_threads = omp_get_max_threads();

    resize(_num_rows, _nnz);
    size = _num_rows;
    nnz = _nnz;

    VNT* col_ids_new = *(&col_ids);
    VNT* row_ids_new = *(&row_ids);
    T* vals_new = *(&vals);

    #pragma omp parallel for schedule(guided, 1024)
    for(VNT row = 0; row < _num_rows; row++)
    {
        for(ENT j = _row_ptr[row]; j < _row_ptr[row + 1]; j++)
        {
            row_ids_new[j] = row;
            col_ids_new[j] = _col_ids[j];
            vals_new[j] = _vals[j];
        }
    }

    bool use_cache_blocking = true;
    if(use_cache_blocking) // do cache blocking optimization with block size equal to L1 or LLC partition
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nnz);
        for(ENT i = 0; i < _nnz; i++)
            sort_indexes[i] = i;

        int seg_size = min((VNT) (512*1024 / sizeof(T)), (size/(num_threads*2)));

        cout << "num segments: " << (size - 1)/seg_size + 1 << endl;

        std::sort(sort_indexes, sort_indexes + _nnz,
                  [row_ids_new, col_ids_new, seg_size](int index1, int index2)
                  {
                      if(col_block<T>(col_ids_new[index1]) == col_block<T>(col_ids_new[index2]))
                          return row_ids_new[index1] < row_ids_new[index2];
                      else
                          return col_block<T>(col_ids_new[index1]) < col_block<T>(col_ids_new[index2]);
                  });
        reorder(row_ids_new, sort_indexes, _nnz);
        reorder(col_ids_new, sort_indexes, _nnz);
        reorder(vals_new, sort_indexes, _nnz);

        MemoryAPI::free_array(sort_indexes);
    }
    else // sort just as in CSR
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nnz);

        #pragma omp parallel for
        for(ENT i = 0; i < _nnz; i++)
            sort_indexes[i] = i;

        std::sort(sort_indexes, sort_indexes + _nnz,
                  [row_ids_new, col_ids_new](int index1, int index2)
                  {
                      if(row_ids_new[index1] == row_ids_new[index2])
                          return col_ids_new[index1] < col_ids_new[index2];
                      else
                          return row_ids_new[index1] < row_ids_new[index2];
                  });
        reorder(row_ids_new, sort_indexes, _nnz);
        reorder(col_ids_new, sort_indexes, _nnz);
        reorder(vals_new, sort_indexes, _nnz);

        MemoryAPI::free_array(sort_indexes);
    }

    // shuffling edges to avoid inter-thread write conflicts by row ids
    for(int tid = 0; tid < num_threads; tid++)
    {
        ENT thread_work_size = (nnz - 1) / num_threads + 1;
        thread_bottom_border[tid] = thread_work_size * tid;
        thread_top_border[tid] = thread_work_size * (tid + 1);
    }

    for(int tid = 0; tid < num_threads; tid++)
    {
        if(tid != (num_threads - 1)) // if not last thread since it does not need to move its top border
        {
            for (ENT i = thread_top_border[tid]; i < nnz - 1; i++)
            {
                VNT cur = row_ids[i];
                VNT next = row_ids[i + 1];
                if (cur == next) // if border on different rows, fix it
                {
                    continue;
                }
                else
                {
                    thread_top_border[tid] = i + 1;
                    thread_bottom_border[tid + 1] = i + 1;
                    break;
                }
            }
        }
    }

    // print thread borders in debug mode
    /*for(int tid = 0; tid < num_threads; tid++)
    {
        cout << "tid " << tid << ") " << thread_bottom_border[tid] << " - " <<  thread_top_border[tid] <<
        "(" << 100.0*(double(thread_top_border[tid] - thread_bottom_border[tid])/nnz) << "%)" << endl;
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
