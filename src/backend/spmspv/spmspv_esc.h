/// @file spmspm_esc.h
/// @author Anton Potapov
/// @version Revision 1.2
/// @brief Parallel map-based SpMSpV with ESC approach
/// @details Implements parallel map-based SpMSpV with ESC approach
/// @date June 13, 2022
#pragma once

/// @brief Parallel map-based SpMSpV with with ESC approach
///
/// This algorithm implements parallel map-based SpMSpV with ESC approach to avoid synchronizations in loop
/// section by accumulating each threads data in it's own hash-map accumulator and then reducing it across the
/// threads into one.
///
/// @param[in] _matrix Pointer to the input matrix
/// @param[in] _x Pointer to the input vector
/// @param[out] _y Pointer to the DenseVector object that will contain the result vector.
/// @param[in] _desc Pointer to a descriptor object
/// @param[in] _accum BinaryOp accumulator
/// @param[in] _op Semiring operation
/// @param[in] _mask SpMSpV mask pointer
/// @see SpMSpV_map_par
/// @see SpMSpV_map_seq
/// @see SpMSpV_map_par_critical
template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV_esc(const MatrixCSR<A> *_matrix,
                             const SparseVector <X> *_x,
                             SparseVector <Y> *_y,
                             Descriptor *_desc,
                             BinaryOpTAccum _accum,
                             SemiringT _op,
                             const Vector <M> *_mask)
{
    LOG_TRACE("Running SpMSpV_esc")
    const X *x_vals = _x->get_vals(); // y is guaranteed to be sparse
    const Index *y_ids = _y->get_ids();

    Y *y_vals = _y->get_vals(); // x is guaranteed to be sparse
    const Index *x_ids = _x->get_ids();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();

    std::unordered_map<VNT, Y> map_output;

    #pragma omp parallel
    {
        VNT vals_per_thread = (x_nvals + omp_get_num_threads() - 1) / omp_get_num_threads();
        int cur_thread_num = omp_get_thread_num();
        VNT val_start_id = cur_thread_num * vals_per_thread;
        VNT val_end_id = (cur_thread_num + 1) * vals_per_thread;
        if (val_start_id > x_nvals) {
            val_start_id = x_nvals;
        }
        if (val_end_id > x_nvals) {
            val_end_id = x_nvals;
        }
        if (val_start_id > val_end_id) {
            val_start_id = val_end_id;
        }
        std::unordered_map<VNT, Y> current_thread_map;
        for (VNT i = val_start_id; i < val_end_id; ++i) {
            VNT ind = x_ids[i];
            X x_val = x_vals[i];
            ENT row_start = _matrix->row_ptr[ind];// this is actually col ptr for mxv operation
            ENT row_end = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++) {
                VNT dest_ind = _matrix->col_ids[j];// this is row_ids
                A mat_val = _matrix->vals[j];
                const auto mul_op_result = mul_op(mat_val, x_val);
                if (current_thread_map.find(dest_ind) == current_thread_map.end())
                    current_thread_map[dest_ind] = add_op(identity_val, mul_op_result);
                else
                    current_thread_map[dest_ind] = add_op(current_thread_map[dest_ind], mul_op_result);
            }
        }
        for (auto [index, val] : current_thread_map) {
            #pragma omp critical
            {
                if (map_output.find(index) == map_output.end())
                    map_output[index] = add_op(identity_val, val);
                else
                    map_output[index] = add_op(map_output[index], val);
            }
        }
    }

    if(_mask != 0) // apply mask and save results
    {
        Desc_value mask_field;
        _desc->get(GrB_MASK, &mask_field);
        if(!_mask->is_dense())
            LOG_DEBUG("warning! costly mask conversion to dense in spmspv esc_based");
        const M *mask_vals = _mask->getDense()->get_vals();
        _y->clear();

        if (mask_field == GrB_STR_COMP) // CMP mask
        {
            for (auto [index, val]: map_output)
            {
                if (mask_vals[index] == 0) // since CMP we keep when 0
                    _y->push_back(index, val);
            }
        }
        else
        {
            for (auto [index, val]: map_output)
            {
                if (mask_vals[index] != 0) // since non-CMP we keep when not 0
                    _y->push_back(index, val);
            }
        }
    }
    else // save results in unmasked case
    {
        _y->clear();
        for (auto [index, val]: map_output)
        {
            _y->push_back(index, val);
        }
    }
}