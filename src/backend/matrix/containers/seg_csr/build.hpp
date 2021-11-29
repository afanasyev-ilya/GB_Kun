/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{


template <typename T>
void MatrixSegmentedCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    size = _size;
    nz = _nz;

    VNT segment_size = 1024 * 1024 / sizeof(int);
    num_segments = (size - 1) / segment_size + 1;
    //num_segments = 4;
    //VNT segment_size = size/num_segments;

    cout << "Using " << num_segments << " segments..." << endl;

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
        int seg_size;
        subgraphs[cur_seg].get_size(&seg_size);
        int seg_nz;
        subgraphs[cur_seg].get_nz(&seg_nz);
        cout << "size (vertices) = " << seg_size << "(" <<
                100.0*(double)seg_size/_size << "%)" << ", nz (edges) = " << seg_nz << " (" <<
                       100.0*(double)seg_nz/_nz << "%) ";
        cout << "avg degree: " << (double)seg_nz / seg_size << endl;
    }

    cout << endl << endl;
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
