/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{


template <typename T>
void MatrixSegmentedCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    size = _size;
    nnz = _nnz;

    cout << size << endl;
    //VNT segment_size = 64 * 1024 / sizeof(T);
    //num_segments = (size - 1) / segment_size + 1;
    num_segments = 12;
    //while(size/num_segments > (1024 * 1024 / sizeof(double)))
    //    num_segments *= 2;
    VNT segment_size = (size - 1)/num_segments + 1;
    cout << "Using " << num_segments << " segments..." << endl;
    cout << "Seg size " << segment_size*sizeof(T)/1e3 << " KB" << endl;

    subgraphs = new SubgraphSegment<T>[num_segments];

    for(ENT i = 0; i < _nnz; i++)
    {
        int seg_id = _col_ids[i]/segment_size;
        subgraphs[seg_id].add_edge(_row_ids[i], _col_ids[i], _vals[i]);
    }

    size_t merge_block_size = 64*1024; // 64 KB
    merge_blocks_number = (size - 1)/merge_block_size + 1;

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
    }

    cout << endl << endl;
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
