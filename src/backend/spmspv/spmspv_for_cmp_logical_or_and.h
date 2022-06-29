#pragma once

template <typename A, typename X, typename Y, typename M>
void SpMSpV_for_cmp_logical_or_and(const MatrixCSR<A> *_matrix,
                                   const SparseVector <X> *_x,
                                   DenseVector <Y> *_y,
                                   const Vector <M> *_mask,
                                   Workspace *_workspace)
{
    LOG_TRACE("Running SpMSpV with functor options (y is dense)")
    Y *prev_y_vals = _y->get_vals();
    Y *old_y_vals = (Y*)_workspace->get_shared_one();
    memcpy(old_y_vals, prev_y_vals, sizeof(Y)*_y->get_size());

    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

    const M *mask_vals = _mask->getDense()->get_vals();

    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < y_size; row++)
        {
            y_vals[row] = 0;
        }

        #pragma omp for
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            if (mask_vals[ind] != 0) {
                continue;
            }
            X x_val = x_vals[i];
            ENT row_start   = _matrix->row_ptr[ind];
            ENT row_end     = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                y_vals[dest_ind] = 1;
            }
        }
    }
}
