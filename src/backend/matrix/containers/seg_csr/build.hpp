/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool custome_compare(const std::pair<int, ENT> &p1, const std::pair<int, ENT> &p2)
{
    return p1.second < p2.second;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double percent_diff(double _val1, double _val2)
{
    double abs_dif = fabs(_val1 - _val2);
    double max_val = max(_val1, _val2);
    return abs_dif / max_val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSegmentedCSR<T>::build(VNT _num_rows,
                                  ENT _nnz,
                                  const ENT *_row_ptr,
                                  const VNT *_col_ids,
                                  const T *_vals,
                                  int _socket)
{
    double t1, t2;
    size = _num_rows;
    nnz = _nnz;

    VNT segment_size = SEG_CSR_CACHE_BLOCK_SIZE / sizeof(T);
    num_segments = (size - 1) / segment_size + 1;
    merge_blocks_number = omp_get_max_threads()*2; // 2 for load balancing
    size_t merge_block_size = (size - 1) / merge_blocks_number + 1;
    while(merge_block_size > (SEG_CSR_MERGE_BLOCK_SIZE/ sizeof(T)))
    {
        merge_blocks_number *= 2;
        merge_block_size = (size - 1) / merge_blocks_number + 1;
    }
    cout << "using " << num_segments << " segments..." << endl;
    cout << "segment size : " << segment_size*sizeof(T)/1e3 << " KB" << endl;
    cout << "merge blocks count : " << merge_blocks_number << endl;
    cout << "merge_block_size : " << merge_block_size*sizeof(T) / 1024 << " KB" << endl;

    // estimate number of edges in each segment
    vector<ENT> estimated_edges_in_segment(num_segments, 0);
    vector<bool> is_small_segment(num_segments, false);
    #pragma omp parallel for schedule(guided, 1024)
    for(VNT row = 0; row < _num_rows; row++) // TODO we have CSC, it must be faster!
    {
        for(ENT j = _row_ptr[row]; j < _row_ptr[row + 1]; j++)
        {
            VNT col = _col_ids[j];
            int seg_id = col/segment_size;
            #pragma omp atomic
            estimated_edges_in_segment[seg_id]++;
        }
    }

    vector<pair<int, ENT>> sorted_segments;
    for(VNT cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        sorted_segments.push_back(make_pair(cur_seg, estimated_edges_in_segment[cur_seg]));
    }
    std::sort( std::begin(sorted_segments), std::end(sorted_segments), custome_compare );

    ENT merged_nnz = 0;
    int merge_segments_num = 0;
    for(int i = 0; i < num_segments; i++)
    {
        int seg_id = sorted_segments[i].first;
        merged_nnz += sorted_segments[i].second;
        is_small_segment[seg_id] = true;
        if(merged_nnz >= 0.2*nnz)
        {
            merge_segments_num = i;
            break;
        }
    }

    vector<int> merge_conversion(num_segments);
    int conversion_pos = 1;
    for(int i = 0; i < num_segments; i++)
    {
        if(is_small_segment[i])
            merge_conversion[i] = 0;
        else
        {
            merge_conversion[i] = conversion_pos;
            conversion_pos += 1;
        }
        cout << "seg " << i << " is supposed to have " << 100.0 * estimated_edges_in_segment[i] / nnz << " % edges, " <<
        " small = " << is_small_segment[i] << " conv indx " << merge_conversion[i] << endl;
    }
    cout << "merging " << merge_segments_num << " smallest from " << num_segments << endl;
    num_segments = (num_segments - merge_segments_num);
    cout << "new segments number: " << num_segments << endl;

    // create segments (must be reworked based on created CSR)
    t1 = omp_get_wtime();
    subgraphs = new SubgraphSegment<T>[num_segments];
    for(VNT row = 0; row < _num_rows; row++)
    {
        for(ENT j = _row_ptr[row]; j < _row_ptr[row + 1]; j++)
        {
            VNT col = _col_ids[j];
            T val = _vals[j];

            int seg_id = col/segment_size;

            if(is_small_segment[seg_id])
            {
                seg_id = 0;
                subgraphs[seg_id].add_edge(row, col, val);
            }
            else
            {
                seg_id = merge_conversion[seg_id];
                subgraphs[seg_id].add_edge(row, col, val);
            }

        }
    }
    t2 = omp_get_wtime();
    cout << "subgraphs adding edges time: " << t2 - t1 << " sec" << endl;

    t1 = omp_get_wtime();
    // construct CSRs and prepare merge blocks
    #pragma omp parallel for schedule(dynamic, 1)
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        subgraphs[cur_seg].construct_csr();
    }
    t2 = omp_get_wtime();
    cout << "converting subgraphs to CSR time (without sort): " << t2 - t1 << " sec" << endl;

    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++) // WARNING! can't be parallel, num-aware alloc inside
    {
        subgraphs[cur_seg].init_buffer_and_copy_edges();
    }

    t1 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1)
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        subgraphs[cur_seg].construct_blocks(merge_blocks_number, merge_block_size);
    }
    t2 = omp_get_wtime();
    cout << "constructing merge blocks time: " << t2 - t1 << " sec" << endl;

    // print stats
    for(int seg_id = 0; seg_id < num_segments; seg_id++)
    {
        cout << "seg " << seg_id<< " stats || ";
        cout << "size (vertices) = " << subgraphs[seg_id].size << "(" <<
             100.0*(double)subgraphs[seg_id].size/size << "%)" << ", nnz (edges) = " << subgraphs[seg_id].nnz << " (" <<
             100.0*(double)subgraphs[seg_id].nnz/nnz << "%) ";
        cout << "avg degree: " << (double)subgraphs[seg_id].nnz / subgraphs[seg_id].size << endl;
        cout << "balancing: " << subgraphs[seg_id].schedule_type << " " << subgraphs[seg_id].load_balanced_type << endl;
        cout << "static ok to use : " << subgraphs[seg_id].static_ok_to_use << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
