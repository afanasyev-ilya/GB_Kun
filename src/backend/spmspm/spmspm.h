#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "spmspm_hash_based_gustavson.h"
#include "spmspm_ijk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VNT spgemm_binary_search(const Index* data, VNT left, VNT right, ENT value)
{
    while (true) {
        if (left > right) {
            return -1;
        }
        VNT mid = left + (right - left) / 2; // div 2 or div rand
        if (data[mid] < value) {
            left = mid + 1;
        } else if (data[mid] > value) {
            right = mid - 1;
        } else {
            return mid;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void balance_matrix_rows(const ENT *_row_ptrs, VNT num_rows, pair<VNT, VNT> *_offsets, int threads_count)
{
    ENT nnz = _row_ptrs[num_rows];
    ENT approx_nnz_per_thread = (nnz - 1) / threads_count + 1;
    for(int tid = 0; tid < threads_count; tid++)
    {
        ENT expected_tid_left_border = approx_nnz_per_thread * tid;
        ENT expected_tid_right_border = approx_nnz_per_thread * (tid + 1);

        const ENT* left_border_ptr = _row_ptrs;
        const ENT* right_border_ptr = _row_ptrs + num_rows;

        auto low_pos = std::lower_bound(left_border_ptr, right_border_ptr, expected_tid_left_border);
        auto up_pos = std::lower_bound(left_border_ptr, right_border_ptr, expected_tid_right_border);

        VNT low_val = low_pos - left_border_ptr;
        VNT up_val = min(num_rows, (VNT) (up_pos - left_border_ptr));

        _offsets[tid] = make_pair(low_val, up_val);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SpMSpM()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMSpM_alloc(Matrix<T> *_matrix_result)
{
    _matrix_result->set_preferred_matrix_format(CSR);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
