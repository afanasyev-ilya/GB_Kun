#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SubgraphSegment
{
    vector<VNT> tmp_row_ids;
    vector<VNT> tmp_col_ids;
    vector<T> tmp_vals;

    VNT size;
    ENT nz;

    double *vertex_buffer;
    VNT *conversion_to_full;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    SubgraphSegment();
    ~SubgraphSegment();

    void add_edge(VNT _row_id, VNT _col_id, T _val);

    void sort_by_row_id();

    void dump();

    void construct_csr();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::add_edge(int _row_id, int _col_id, T _val)
{
    tmp_row_ids.push_back(_row_id);
    tmp_col_ids.push_back(_col_id);
    tmp_vals.push_back(_val);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::sort_by_row_id()
{
    ENT nz = tmp_col_ids.size();
    ENT *sort_indexes;
    MemoryAPI::allocate_array(&sort_indexes, nz);
    for(ENT i = 0; i < nz; i++)
        sort_indexes[i] = i;

    VNT *tmp_ptr = tmp_row_ids.data();

    std::sort(sort_indexes, sort_indexes + nz,
              [tmp_ptr](int index1, int index2)
              {
                  return tmp_ptr[index1] < tmp_ptr[index2];
              });

    reorder(tmp_row_ids.data(), sort_indexes, nz);
    reorder(tmp_col_ids.data(), sort_indexes, nz);
    reorder(tmp_vals.data(), sort_indexes, nz);

    MemoryAPI::free_array(sort_indexes);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::construct_csr()
{
    size = 1;
    nz = tmp_row_ids.size();

    map<VNT, VNT> conv;
    conv[tmp_row_ids[0]] = 0;

    VNT cur_row = tmp_row_ids[0];
    for(ENT i = 1; i < nz; i++)
    {
        if(tmp_row_ids[i] != cur_row)
        {
            conv[tmp_row_ids[i]] = cur_row;
            cur_row = tmp_row_ids[i];
            size++;
        }
    }

    MemoryAPI::allocate_array(&row_ptr, size + 1);
    MemoryAPI::allocate_array(&col_ids, nz);
    MemoryAPI::allocate_array(&vals, nz);
    MemoryAPI::allocate_array(&conversion_to_full, size);
    MemoryAPI::allocate_array(&vertex_buffer, size);

    for(VNT i = 0; i < size + 1; i++)
        row_ptr[i] = 0;

    for(VNT i = 0; i < size; i++)
        conversion_to_full[i] = i;

    for (ENT i = 0; i < nz; i++)
    {
        row_ptr[conv[tmp_row_ids[i]] + 1]++;

        VNT row_in_full = tmp_row_ids[i];
        VNT row_in_seg = conv[tmp_row_ids[i]];

        //conversion_to_full[row_in_seg] = row_in_full;
    }

    for (VNT i = 0; i < size; i++)
        row_ptr[i + 1] += row_ptr[i];

    for (ENT i = 0; i < nz; i++)
    {
        col_ids[i] = tmp_col_ids[i];
        vals[i] = tmp_vals[i];
    }

    /*cout << "size: " << size << endl;
    cout << "nz: " << nz << endl;
    for(VNT i = 0; i < size; i++)
    {
        cout << "[" << row_ptr[i] << " - " << row_ptr[i + 1] << "], ";
    }
    cout << endl;

    for(ENT i = 0; i < nz; i++)
    {
        cout << col_ids[i] << " ";
    }
    cout << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::dump()
{
    for(ENT i = 0; i < tmp_row_ids.size(); i++)
    {
        cout << "(" << tmp_row_ids[i] << ", " << tmp_col_ids[i] << "), ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
SubgraphSegment<T>::SubgraphSegment()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
SubgraphSegment<T>::~SubgraphSegment()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
    MemoryAPI::free_array(vertex_buffer);
    MemoryAPI::free_array(conversion_to_full);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


