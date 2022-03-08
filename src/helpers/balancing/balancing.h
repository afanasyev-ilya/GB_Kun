#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void balance_matrix_rows(const vector<ENT> &_row_ptrs, vector<pair<VNT, VNT>> &_offsets)
{
    ENT nnz = _row_ptrs.back();
    int threads_count = omp_get_max_threads();
    ENT approx_nnz_per_thread = (nnz - 1) / threads_count + 1;
    for(int tid = 0; tid < threads_count; tid++)
    {
        ENT expected_tid_left_border = approx_nnz_per_thread * tid;
        ENT expected_tid_right_border = approx_nnz_per_thread * (tid + 1);
        //cout << "expected borders: " << expected_tid_left_border << " " << expected_tid_right_border << endl;

        auto low_pos = std::lower_bound(_row_ptrs.begin(), _row_ptrs.end(), expected_tid_left_border);
        auto up_pos = std::lower_bound(_row_ptrs.begin(), _row_ptrs.end(), expected_tid_right_border);

        //cout << "tid: " << tid << " " << *low_pos << " | " << *up_pos << " processing " <<
        //    100.0*(*up_pos - *low_pos)/nnz << "% elements" << endl;

        _offsets.push_back(make_pair<VNT, VNT>(low_pos - _row_ptrs.begin(), up_pos - _row_ptrs.begin()));
    }

    /*for(auto i: _offsets)
    {
        cout << i.first << " - " << i.second << endl;
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
