/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
namespace backend{

bool custome_compare(const std::pair<int, ENT> &p1, const std::pair<int, ENT> &p2)
{
    return p1.second < p2.second;
}

template <typename T>
void MatrixSegmentedCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _socket)
{
    size = _size;
    nnz = _nnz;

    cout << size << endl;
    VNT segment_size = 512 * 1024 / sizeof(T);
    num_segments = (size - 1) / segment_size + 1;
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

    // construct CSRs and prepare merge blocks
    double avg_avg_degree = 0;
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        subgraphs[cur_seg].sort_by_row_id();
        subgraphs[cur_seg].construct_csr();
        subgraphs[cur_seg].construct_blocks(merge_blocks_number, merge_block_size);
        sorted_segments.push_back(make_pair(cur_seg, subgraphs[cur_seg].nnz));
        avg_avg_degree += ((double)subgraphs[cur_seg].nnz / subgraphs[cur_seg].size)/num_segments;
    }

    // do load balancing optimization for the largest segment
    std::sort( std::begin(sorted_segments), std::end(sorted_segments), custome_compare );

    ENT sum_edges = 0;
    load_balanced_threshold = num_segments;
    int cnt = 0;
    for(int seg_idx = num_segments - 1; seg_idx >= 0; seg_idx--)
    {
        int seg_id = sorted_segments[seg_idx].first;

        if(cnt < 1) // 25% of total edges
        {
            subgraphs[seg_id].construct_load_balancing();
            load_balanced_threshold = seg_idx;
        }
        else
            break;
        sum_edges += subgraphs[seg_id].nnz;
        cnt++;
    }

    cout << "load balanced threshold : " << load_balanced_threshold << " / " << num_segments << endl;

    cout << "avg graph degree: " << ((double)_nnz)/_size << endl;
    cout << "avg avg degree: " << avg_avg_degree << endl;
    schedule_type.resize(num_segments);
    load_balanced_type.resize(num_segments);
    for(int cur_seg = 0; cur_seg < num_segments; cur_seg++)
    {
        int seg_id = sorted_segments[cur_seg].first;
        cout << "seg " << seg_id<< " stats || ";
        cout << "size (vertices) = " << subgraphs[seg_id].size << "(" <<
             100.0*(double)subgraphs[seg_id].size/_size << "%)" << ", nnz (edges) = " << subgraphs[seg_id].nnz << " (" <<
             100.0*(double)subgraphs[seg_id].nnz/_nnz << "%) ";
        cout << "avg degree: " << (double)subgraphs[seg_id].nnz / subgraphs[seg_id].size << endl;

        // static when better? ru - avg_deg = avg_avg_deg
        // static is better when avg_degree is very low

        // load balancing when better? when degree is very large (> threshold)

        // if ave_degree is low or == to avg avg - static
        // else -- guided

        // if % of nnz in seg is high -- load balancied groups (must be corrected since avg degree is much smaller!)
        // try without load balanced groups first
    }
}
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
