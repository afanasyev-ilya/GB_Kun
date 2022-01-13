/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{

bool custome_compare(const std::pair<int, ENT> &p1, const std::pair<int, ENT> &p2)
{
    return p1.second > p2.second;
}

template <typename T>
void MatrixSegmentedCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    size = _size;
    nnz = _nnz;

    cout << size << endl;
    VNT segment_size = 512 * 1024 / sizeof(T);
    num_segments = (size - 1) / segment_size + 1;
    //num_segments = (omp_get_max_threads())*4; // since 4x for load balancing
    //VNT segment_size = (size - 1)/num_segments + 1;
    cout << "Using " << num_segments << " segments..." << endl;
    cout << "Seg size " << segment_size*sizeof(T)/1e3 << " KB" << endl;

    // create segments (must be reworked based on created CSR)
    subgraphs = new SubgraphSegment<T>[num_segments];
    for(ENT i = 0; i < _nnz; i++)
    {
        int seg_id = _col_ids[i]/segment_size;
        subgraphs[seg_id].add_edge(_row_ids[i], _col_ids[i], _vals[i]);
    }

    /*size_t merge_block_size = 16*1024; // 64 KB
    merge_blocks_number = (size - 1)/merge_block_size + 1;*/
    merge_blocks_number = omp_get_max_threads()*2; // 2 for load balancing
    size_t merge_block_size = (_size - 1) / merge_blocks_number + 1;
    while(merge_block_size > (32*1024/ sizeof(T)))
    {
        merge_blocks_number *= 2;
        merge_block_size = (_size - 1) / merge_blocks_number + 1;
    }

    cout << "merge blocks count : " << merge_blocks_number << endl;
    cout << "merge_block_size : " << merge_block_size*sizeof(T) / 1024 << " KB" << endl;

    // prepare merge blocks
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        subgraphs[cur_seg].sort_by_row_id();
        subgraphs[cur_seg].construct_csr();
        subgraphs[cur_seg].construct_blocks(merge_blocks_number, merge_block_size);
        cout << "seg " << cur_seg << " stats || ";
        cout << "size (vertices) = " << subgraphs[cur_seg].size << "(" <<
             100.0*(double)subgraphs[cur_seg].size/_size << "%)" << ", nnz (edges) = " << subgraphs[cur_seg].nnz << " (" <<
             100.0*(double)subgraphs[cur_seg].nnz/_nnz << "%) ";
        cout << "avg degree: " << (double)subgraphs[cur_seg].nnz / subgraphs[cur_seg].size << endl;
        sorted_segments.push_back(make_pair(cur_seg, subgraphs[cur_seg].nnz));
    }

    // do load balancing optimization for the largest segment
    std::sort( std::begin(sorted_segments), std::end(sorted_segments), custome_compare );
    int largest_segment = sorted_segments[0].first;
    subgraphs[largest_segment].construct_load_balancing();
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
