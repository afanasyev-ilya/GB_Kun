/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{


template <typename T>
void MatrixSegmentedCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    size = _size;
    nz = _nz;

    cout << size << endl;
    VNT segment_size = 512 * 1024 / sizeof(T);
    num_segments = (size - 1) / segment_size + 1;
    //num_segments = 256;
    //while(size/num_segments > (1024 * 1024 / sizeof(double)))
    //    num_segments *= 2;
    //VNT segment_size = size/num_segments;
    cout << "Using " << num_segments << " segments..." << endl;
    cout << "Seg size " << segment_size*sizeof(T)/1e3 << " KB" << endl;

    subgraphs = new SubgraphSegment<T>[num_segments];

    for(ENT i = 0; i < _nz; i++)
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
             100.0*(double)subgraphs[cur_seg].size/_size << "%)" << ", nz (edges) = " << subgraphs[cur_seg].nz << " (" <<
             100.0*(double)subgraphs[cur_seg].nz/_nz << "%) ";
        cout << "avg degree: " << (double)subgraphs[cur_seg].nz / subgraphs[cur_seg].size << endl;
    }

    cout << endl << endl;
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
