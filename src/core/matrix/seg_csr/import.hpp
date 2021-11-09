/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSegmentedCSR<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz)
{
    size = _size;
    nz = _nz;

    //VNT segment_size = 1024 * 1024 / sizeof(int);
    //num_segments = (size - 1) / segment_size + 1;
    num_segments = 2;
    VNT segment_size = size/num_segments;

    subgraphs = new SubgraphSegment<T>[num_segments];

    for(ENT i = 0; i < _nz; i++)
    {
        int seg_id = _col_ids[i]/segment_size;
        subgraphs[seg_id].add_edge(_row_ids[i], _col_ids[i], _vals[i]);
    }

    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        //cout << "seg " << cur_seg << endl;
        subgraphs[cur_seg].sort_by_row_id();
        //subgraphs[cur_seg].dump();
        subgraphs[cur_seg].construct_csr();
        //cout << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
