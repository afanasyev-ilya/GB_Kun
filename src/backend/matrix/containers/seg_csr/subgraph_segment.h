#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas{
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ScheduleType
{
    STATIC = 0,
    GUIDED = 1
};

enum LoadBalancedType
{
    ONE_GROUP = 0,
    MANY_GROUPS = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class MatrixSegmentedCSR;

template <typename T>
class SubgraphSegment
{
public:
    SubgraphSegment();
    ~SubgraphSegment();

    void add_edge(VNT _row_id, VNT _col_id, T _val);

    void dump();

    void construct_csr();

    void construct_blocks(VNT _block_number, size_t _block_size);

    void construct_load_balancing();
private:
    vector<VNT> tmp_row_ids;
    vector<VNT> tmp_col_ids;
    vector<T> tmp_vals;

    VNT size;
    ENT nnz;

    double *vertex_buffer;
    VNT *conversion_to_full;

    ENT *row_ptr;
    T *vals;
    VNT *col_ids;

    VNT *block_starts;
    VNT *block_ends;

    VNT first_col, last_col;

    static const int vg_num = 6; // 9 is best currently
    VertexGroup vertex_groups[vg_num];

    ScheduleType schedule_type;
    LoadBalancedType load_balanced_type;

    template <typename A, typename X, typename Y, typename BinaryOpTAccum, typename SemiringT>
    friend void SpMV(const MatrixSegmentedCSR<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Workspace *_workspace);

    template <typename Y>
    friend class MatrixSegmentedCSR;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::add_edge(VNT _row_id, VNT _col_id, T _val)
{
    tmp_row_ids.push_back(_row_id);
    tmp_col_ids.push_back(_col_id);
    tmp_vals.push_back(_val);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::construct_csr()
{
    nnz = tmp_row_ids.size();

    double t1 = omp_get_wtime();
    size = 0;
    map<VNT, VNT> conv;
    for(ENT i = 0; i < nnz; i++)
    {
        if(conv.find(tmp_row_ids[i]) == conv.end())
        {
            conv[tmp_row_ids[i]] = size;
            size++;
        }
    }
    double t2 = omp_get_wtime();
    cout << "map creation time: " << (t2 - t1) << " sec" << endl;

    MemoryAPI::allocate_array(&row_ptr, size + 1);
    MemoryAPI::allocate_array(&col_ids, nnz);
    MemoryAPI::allocate_array(&vals, nnz);
    MemoryAPI::allocate_array(&conversion_to_full, size);
    MemoryAPI::allocate_array(&vertex_buffer, size);

    #pragma omp parallel for schedule(static) // cache-aware alloc
    for(VNT i = 0; i < size; i++)
        vertex_buffer[i] = 0;

    for(VNT i = 0; i < size + 1; i++)
        row_ptr[i] = 0;

    for(VNT i = 0; i < size; i++)
        conversion_to_full[i] = 0;

    t1 = omp_get_wtime();
    for (ENT i = 0; i < nnz; i++)
    {
        row_ptr[conv[tmp_row_ids[i]] + 1]++;

        VNT row_in_full = tmp_row_ids[i];
        VNT row_in_seg = conv[tmp_row_ids[i]];

        conversion_to_full[row_in_seg] = row_in_full;
    }
    t2 = omp_get_wtime();
    cout << "row ptr cnt time: " << (t2 - t1) << " sec" << endl;

    for (VNT i = 0; i < size; i++)
        row_ptr[i + 1] += row_ptr[i];

    first_col = size;
    last_col = 0;
    for (ENT i = 0; i < nnz; i++)
    {
        col_ids[i] = tmp_col_ids[i];
        vals[i] = tmp_vals[i];
        first_col = min(first_col, col_ids[i]);
        last_col = max(last_col, col_ids[i]);
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

    for(VNT i = 0; i < _block_number; i++)
    {
        block_starts[i] = -1;
        block_ends[i] = -1;
    }

    block_starts[block_nums[0]] = 0;
    for(VNT i = 0; i < size - 1; i++)
    {
        VNT first = block_nums[i];
        VNT second = block_nums[i + 1];

        if(first != second)
        {
            block_ends[first] = i + 1;
            block_starts[second] = i + 1;
        }
    }
    block_ends[block_nums[size - 1]] = size;

    for(VNT i = 0; i < _block_number; i++)
    {
        if(block_starts[i] == -1) // is unset
            block_starts[i] = block_ends[i];
        if(block_ends[i] == -1) // is unset
            block_ends[i] = block_starts[i];
    }

    /*for(VNT bl = 0; bl < _block_number; bl++)
    {
        VNT min_v = INT_MAX, max_v = 0;

        for(VNT i = block_starts[bl]; i < block_ends[bl]; i++)
        {
            VNT loc_v = i;
            VNT rem_v = conversion_to_full[loc_v];
            min_v = min(min_v, rem_v);
            max_v = max(max_v, rem_v);
        }
        VNT size_v = max_v - min_v;
        if(block_starts[bl] != block_ends[bl])
        {
            cout << "block: " << bl << " size " << size_v * sizeof(T) / 1e3 << " KB" << endl;
        }
    }*/
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void SubgraphSegment<T>::construct_load_balancing()
{
    ENT step = 4;
    ENT first = 4;
    vertex_groups[0].set_thresholds(0, first);
    for(int i = 1; i < (vg_num - 1); i++)
    {
        vertex_groups[i].set_thresholds(first, first*step);
        first *= step;
    }
    vertex_groups[vg_num - 1].set_thresholds(first, INT_MAX);

    for(VNT row = 0; row < size; row++)
    {
        ENT connections_count = row_ptr[row + 1] - row_ptr[row];
        for(int vg = 0; vg < vg_num; vg++)
            if(vertex_groups[vg].in_range(connections_count))
                vertex_groups[vg].push_back(row);
    }

    for(int i = 0; i < vg_num; i++)
    {
        vertex_groups[i].finalize_creation(0); // TODO target socket of graph
    }
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


