#pragma once

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
void EdgeListContainer<T>::save_as_mtx(string _file_name)
{
    VNT* src_ids_new = &src_ids[0];
    VNT* dst_ids_new = &dst_ids[0];
    T* vals_new = &edge_vals[0];

    ENT *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, edges_count);

    #pragma omp parallel for
    for(ENT i = 0; i < edges_count; i++)
        sort_indexes[i] = i;

    std::sort(sort_indexes, sort_indexes + edges_count,
              [src_ids_new, dst_ids_new](int index1, int index2)
              {
                  if(src_ids_new[index1] == src_ids_new[index2])
                      return dst_ids_new[index1] < dst_ids_new[index2];
                  else
                      return src_ids_new[index1] < src_ids_new[index2];
              });
    reorder(src_ids_new, sort_indexes, edges_count);
    reorder(dst_ids_new, sort_indexes, edges_count);
    reorder(vals_new, sort_indexes, edges_count);

    MemoryAPI::free_array(sort_indexes);

    ENT unique_edges = 0;
    for(ENT i = 1; i < edges_count; i++)
        if((src_ids_new[i] != src_ids_new[i - 1]) && (dst_ids_new[i] != dst_ids_new[i - 1]) && (dst_ids_new[i] != src_ids_new[i]))
            unique_edges++;

    ofstream matrix_file;
    matrix_file.open (_file_name);
    matrix_file << "%%MatrixMarket matrix coordinate pattern general" << endl;
    matrix_file << vertices_count << " " << vertices_count << " " << unique_edges << endl;
    for(ENT i = 1; i < edges_count; i++)
    {
        if((src_ids_new[i] != src_ids_new[i - 1]) && (dst_ids_new[i] != dst_ids_new[i - 1]) && (src_ids_new[i] != dst_ids_new[i]))
            matrix_file << src_ids_new[i] + 1 << " " << dst_ids_new[i] + 1 << endl;
    }
    matrix_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
