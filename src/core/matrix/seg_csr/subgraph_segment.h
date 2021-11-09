#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SubgraphSegment
{
private:
    vector<VNT> tmp_row_ids;
    vector<VNT> tmp_col_ids;
    vector<T> tmp_vals;

    VNT size;
    ENT nz;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;
public:

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


