#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCOO<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    int num_threads = omp_get_max_threads();

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
    if(use_cache_blocking) // do cache blocking optimization with block size equal to L1 or LLC partition
    {
        ENT *sort_indexes;
        MemoryAPI::allocate_array(&sort_indexes, _nnz);

        #pragma omp parallel for
        for(ENT i = 0; i < _nnz; i++)
            sort_indexes[i] = i;

        //int seg_size = min((VNT) (512*1024 / sizeof(T)), (size/(num_threads*2)));
        int seg_size = 512*1024 / sizeof(T);

        cout << "num segments: " << (size - 1)/seg_size + 1 << endl;

        std::sort(sort_indexes, sort_indexes + _nnz,
                  [_row_ids, _col_ids, seg_size](int index1, int index2)
                  {
                          if(_row_ids[index1] / seg_size == _row_ids[index2] / seg_size)
                              if(_col_ids[index1] / seg_size == _col_ids[index2] / seg_size)
                                  return _row_ids[index1] < _row_ids[index2];
                              else
                                return _col_ids[index1] / seg_size < _col_ids[index2] / seg_size;
                          else
                              return _row_ids[index1] / seg_size < _row_ids[index2] / seg_size;
                  });
        cout << "sort done " << endl;
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
                  [_row_ids, _col_ids](int index1, int index2)
                  {
                      if(_row_ids[index1] == _row_ids[index2])
                          return _col_ids[index1] < _col_ids[index2];
                      else
                          return _row_ids[index1] < _row_ids[index2];
                  });
        reorder(row_ids_new, sort_indexes, _nnz);
        reorder(col_ids_new, sort_indexes, _nnz);
        reorder(vals_new, sort_indexes, _nnz);

        MemoryAPI::free_array(sort_indexes);
    }

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

    for(int tid = 0; tid < num_threads; tid++)
    {
        cout << "tid " << tid << ") " << thread_bottom_border[tid] << " - " <<  thread_top_border[tid] <<
        "(" << 100.0*(double(thread_top_border[tid] - thread_bottom_border[tid])/nnz) << "%)" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
