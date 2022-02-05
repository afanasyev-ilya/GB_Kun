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
    cout << "Using " << num_segments << " segments..." << endl;
    cout << "Seg size " << segment_size*sizeof(T)/1e3 << " KB" << endl;

    merge_blocks_number = omp_get_max_threads()*2; // 2 for load balancing
    size_t merge_block_size = (size - 1) / merge_blocks_number + 1;
    while(merge_block_size > (SEG_CSR_MERGE_BLOCK_SIZE/ sizeof(T)))
    {
        merge_blocks_number *= 2;
        merge_block_size = (size - 1) / merge_blocks_number + 1;
    }

    cout << "merge blocks count : " << merge_blocks_number << endl;
    cout << "merge_block_size : " << merge_block_size*sizeof(T) / 1024 << " KB" << endl;

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
            subgraphs[seg_id].add_edge(row, col, val);
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

    double avg_avg_degree = 0;
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        sorted_segments.push_back(make_pair(cur_seg, subgraphs[cur_seg].nnz));
        avg_avg_degree += ((double) subgraphs[cur_seg].nnz / subgraphs[cur_seg].size) / num_segments;
    }

    t1 = omp_get_wtime();
    // do load balancing optimization for the largest segment
    std::sort( std::begin(sorted_segments), std::end(sorted_segments), custome_compare );

    cout << "avg graph degree: " << ((double)nnz)/size << endl;
    cout << "avg avg degree: " << avg_avg_degree << endl;
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        int seg_id = sorted_segments[cur_seg].first;

        double seg_avg_degree = (double)subgraphs[seg_id].nnz / subgraphs[seg_id].size;
        if(seg_avg_degree < 0.9*avg_avg_degree || percent_diff(seg_avg_degree, avg_avg_degree) < 0.1)
            subgraphs[seg_id].schedule_type = STATIC;
        else
            subgraphs[seg_id].schedule_type = GUIDED;

        if(subgraphs[seg_id].nnz > 0.15*nnz)
        {
            subgraphs[seg_id].load_balanced_type = MANY_GROUPS;
            subgraphs[seg_id].construct_load_balancing();
        }
        else
            subgraphs[seg_id].load_balanced_type = ONE_GROUP;

        cout << "seg " << seg_id<< " stats || ";
        cout << "size (vertices) = " << subgraphs[seg_id].size << "(" <<
             100.0*(double)subgraphs[seg_id].size/size << "%)" << ", nnz (edges) = " << subgraphs[seg_id].nnz << " (" <<
             100.0*(double)subgraphs[seg_id].nnz/nnz << "%) ";
        cout << "avg degree: " << (double)subgraphs[seg_id].nnz / subgraphs[seg_id].size << endl;
        cout << "balancing: " << subgraphs[seg_id].schedule_type << " " << subgraphs[seg_id].load_balanced_type << endl;
    }
    t2 = omp_get_wtime();
    cout << "doing load balancing time: " << t2 - t1 << " sec" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
