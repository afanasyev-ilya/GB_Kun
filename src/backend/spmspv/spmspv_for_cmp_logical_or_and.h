#pragma once

template <typename A, typename X, typename Y, typename M>
void SpMSpV_for_cmp_logical_or_and(const MatrixCSR<A> *_matrix,
                                   const SparseVector <X> *_x,
                                   DenseVector <Y> *_y,
                                   const Vector <M> *_mask,
                                   Workspace *_workspace)
{
    LOG_TRACE("Running SpMSpV with functor options (y is dense)")
    const X *x_vals = _x->get_vals();
    const Index *x_ids = _x->get_ids();
    Y *y_vals = _y->get_vals();

    VNT x_nvals = _x->get_nvals();
    VNT y_size = _y->get_size();

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
            X x_val = x_vals[i];
            ENT row_start = _matrix->row_ptr[ind];
            ENT row_end = _matrix->row_ptr[ind + 1];

            for (ENT j = row_start; j < row_end; j++)
            {
                VNT dest_ind = _matrix->col_ids[j];
                A dest_val = _matrix->vals[j];

                y_vals[dest_ind] = 1;
            }
        }
    }

    if(_mask->is_dense())
    {
        const M *mask_vals = _mask->getDense()->get_vals();
        #pragma omp parallel for
        for (VNT i = 0; i < _mask->get_size(); i++)
        {
            if(mask_vals[i] == 0) // == 0 since CMP mask
                y_vals[i] = y_vals[i];
            else
                y_vals[i] = 0;
        }
    }
    else
    {
        bool *dense_mask = (bool*)_workspace->get_mask_conversion();

        const VNT mask_nvals = _mask->getSparse()->get_nvals();
        const VNT *mask_ids = _mask->getSparse()->get_ids();
        #pragma omp parallel
        {
            #pragma omp for
            for (VNT i = 0; i < _mask->get_size(); i++)
                dense_mask[i] = 1;

            #pragma omp for
            for (VNT i = 0; i < mask_nvals; i++)
            {
                VNT mask_id = mask_ids[i];
                dense_mask[mask_id] = 0; // we deactivate all values from original mask since this is CMP
            }

            #pragma omp for
            for (VNT i = 0; i < _mask->get_size(); i++)
            {
                if(dense_mask[i] == 1)
                    y_vals[i] = y_vals[i];
                else
                    y_vals[i] = 0;
            }
        }
    }

}
