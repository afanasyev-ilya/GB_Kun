#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
    namespace backend {


template <typename T>
class SubgraphSegment
{
public:
    SubgraphSegment();
    ~SubgraphSegment();

    void add_edge(VNT _row_id, VNT _col_id, T _val);

    void sort_by_row_id();

    void dump();

    void construct_csr();

    void get_size(VNT* _size) const{
        *_size = size;
    }
    void get_nz(VNT* _nz) const{
        *_nz = nz;
    }
    ENT* get_row() {
        return row_ptr;
    };

    const ENT* get_row() const {
        return row_ptr;
    };

    VNT* get_col() {
        return col_ids;
    };

    const VNT* get_col() const {
        return col_ids;
    };

    T* get_vals() {
        return vals;
    };

    const T* get_vals() const {
        return vals;
    };

    VNT* get_block_start() {
        return block_starts;
    };

    const VNT* get_block_start() const {
        return block_starts;
    };

    VNT* get_block_end() {
        return block_ends;
    };

    const VNT* get_block_end() const {
        return block_ends;
    };

    double* get_vbuffer() {
        return vertex_buffer;
    };

    const double* get_vbuffer() const {
        return vertex_buffer;
    };

    VNT* get_conversion() {
        return conversion_to_full;
    };

    const VNT* get_conversion() const {
        return conversion_to_full;
    };

    void construct_blocks(VNT _block_number, size_t _block_size);

private:
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

    VNT *block_starts;
    VNT *block_ends;

//    template <typename Y>
//    friend void SpMV(MatrixSegmentedCSR<Y> &_matrix, DenseVector<Y> &_x, DenseVector<Y> &_y);

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
    nz = tmp_row_ids.size();

    size = 0;
    map<VNT, VNT> conv;
    for(ENT i = 0; i < nz; i++)
    {
        if(conv.find(tmp_row_ids[i]) == conv.end())
        {
            conv[tmp_row_ids[i]] = size;
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
        conversion_to_full[i] = 0;

    for (ENT i = 0; i < nz; i++)
    {
        row_ptr[conv[tmp_row_ids[i]] + 1]++;

        VNT row_in_full = tmp_row_ids[i];
        VNT row_in_seg = conv[tmp_row_ids[i]];

        conversion_to_full[row_in_seg] = row_in_full;
    }

    for (VNT i = 0; i < size; i++)
        row_ptr[i + 1] += row_ptr[i];

    for (ENT i = 0; i < nz; i++)
    {
        col_ids[i] = tmp_col_ids[i];
        vals[i] = tmp_vals[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::construct_blocks(VNT _block_number, size_t _block_size)
{
    MemoryAPI::allocate_array(&block_starts, _block_number);
    MemoryAPI::allocate_array(&block_ends, _block_number);
    vector<VNT> block_nums(size);

    for(VNT i = 0; i < size; i++)
    {
        VNT large_graph_vertex = conversion_to_full[i];
        block_nums[i] = large_graph_vertex / _block_size;
    }

    block_starts[block_nums[0]] = 0;

    for(VNT i = 1; i < size - 1; i++)
    {
        VNT prev_block = block_nums[i - 1];
        VNT cur_block = block_nums[i];
        VNT next_block = block_nums[i + 1];

        if(cur_block != prev_block)
            block_starts[cur_block] = i;

        if(cur_block != next_block)
            block_ends[cur_block] = i + 1;
    }

    block_ends[block_nums[size - 1]] = size;
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
    MemoryAPI::free_array(block_starts);
    MemoryAPI::free_array(block_ends);
}
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


