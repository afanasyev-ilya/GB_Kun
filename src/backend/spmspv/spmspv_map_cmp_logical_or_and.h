/// @file spmspm_esc.h
/// @author Anton Potapov
/// @version Revision 1.2
/// @brief BFS-optimized sequential map-based SpMSpV
/// @details Implements COMP-masked sequential map-based SpMSpV with LogicalOrAnd semiring operation and no accumulator
/// @date June 13, 2022

#pragma once

/// @brief BFS-optimized sequential map-based SpMSpV
///
/// This algorithm is an BFS-optimized version of sequential map-based SpMSpV algorithm that assumes
/// usage of complimentary mask, LogicalOrAnd semiring operation and no accumulator.
///
/// @param[in] _matrix Pointer to the input matrix
/// @param[in] _x Pointer to the input vector
/// @param[out] _y Pointer to the DenseVector object that will contain the result vector.
/// @param[in] _mask SpMSpV mask pointer
/// @see SpMSpV_for_cmp_logical_or_and
template <typename A, typename X, typename Y, typename M>
void SpMSpV_map_cmp_logical_or_and(const MatrixCSR<A> *_matrix,
                                   const SparseVector <X> *_x,
                                   SparseVector <Y> *_y,
                                   const Vector <M> *_mask)
{
    LOG_TRACE("Running SpMSpV_map_seq")
    const X *x_vals = _x->get_vals(); // y is guaranteed to be sparse
    const Index *y_ids = _y->get_ids();

    Y *y_vals = _y->get_vals(); // x is guaranteed to be sparse
    const Index *x_ids = _x->get_ids();

    auto identity_val = 1;

    VNT x_nvals = _x->get_nvals();

    std::unordered_map<VNT, Y> map_output;

    const M *mask_vals = _mask->getDense()->get_vals();

    for (VNT i = 0; i < x_nvals; i++)
    {
        VNT ind = x_ids[i];
        X x_val = x_vals[i];
        ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
        ENT row_end   = _matrix->row_ptr[ind + 1];

        for (ENT j = row_start; j < row_end; j++)
        {
            VNT dest_ind = _matrix->col_ids[j]; // this is row_ids

            if (mask_vals[dest_ind] != 0) {
                continue;
            }

            A mat_val = _matrix->vals[j];

            if(map_output.find(dest_ind) == map_output.end())
                map_output[dest_ind] = (mat_val && x_val);
            else
                map_output[dest_ind] = map_output[dest_ind] || (mat_val && x_val);
        }
    }

    _y->clear();
    for (auto [index, val]: map_output)
    {
        _y->push_back(index, val);
    }
}