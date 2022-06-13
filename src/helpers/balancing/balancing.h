/**
  @file balancing.h
  @author Lastname:Firstname:A00123456:cscxxxxx
  @version Revision 1.1
  @date June 10, 2022
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * balance_matrix_rows function.
 * @brief The function balances matrix rows
 * @param _row_ptrs rows
 * @param _offsets offsets
*/


void balance_matrix_rows(const vector<ENT> &_row_ptrs, vector<pair<VNT, VNT>> &_offsets)
{
    VNT nrows = _row_ptrs.size() - 1;
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

        VNT low_val = low_pos - _row_ptrs.begin();
        VNT up_val = min(nrows, (VNT)(up_pos - _row_ptrs.begin()));

        _offsets.emplace_back(low_val, up_val);
    }

    /*for(auto i: _offsets)
    {
        cout << i.first << " - " << i.second << endl;
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
