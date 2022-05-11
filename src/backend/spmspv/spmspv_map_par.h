#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_TBB__
template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV_map_par(const MatrixCSR<A> *_matrix,
                    const SparseVector <X> *_x,
                    SparseVector <Y> *_y,
                    Descriptor *_desc,
                    BinaryOpTAccum _accum,
                    SemiringT _op,
                    const Vector <M> *_mask,
                    Workspace *_workspace)
{
    const X *x_vals = _x->get_vals(); // y is guaranteed to be sparse
    const Index *y_ids = _y->get_ids();

    Y *y_vals = _y->get_vals(); // x is guaranteed to be sparse
    const Index *x_ids = _x->get_ids();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();

    tbb::concurrent_hash_map<VNT, Y> map_output;
    bool static_ok_to_use = true;
    int total_threads = omp_get_max_threads();
    ENT total_edges = 0;
    const int max_threads = 128;
    VNT sum_array[max_threads];
    VNT *search_array = (VNT*)_workspace->get_shared_one();
    #pragma omp parallel
    {
        ENT processed_edges = 0;
        #pragma omp for schedule(static)
        for (VNT i = 0; i < x_nvals; i++)
        {
            VNT ind = x_ids[i];
            ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
            ENT row_end   = _matrix->row_ptr[ind + 1];
            processed_edges += row_end - row_start;
        }

        #pragma omp atomic
        total_edges += processed_edges;

        #pragma omp barrier

        double real_percent = 100.0*((double)processed_edges/total_edges);
        double supposed_percent = 100.0/total_threads;

        if(fabs(real_percent - supposed_percent) > 4) // if difference is more than 4%, static not ok to use
            static_ok_to_use = false;

        #pragma omp barrier

        if(static_ok_to_use)
        {
            #pragma omp for schedule(static)
            for (VNT i = 0; i < x_nvals; i++)
            {
                VNT ind = x_ids[i];
                X x_val = x_vals[i];
                ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
                ENT row_end   = _matrix->row_ptr[ind + 1];
                typename tbb::concurrent_hash_map<VNT, Y>::accessor a;

                for (ENT j = row_start; j < row_end; j++)
                {
                    VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
                    A mat_val = _matrix->vals[j];

                    if(!map_output.find(a, dest_ind)) {
                        map_output.insert(a, dest_ind);
                        a->second = add_op(identity_val, mul_op(mat_val, x_val));
                    } else {
                        a->second = add_op(a->second, mul_op(mat_val, x_val));
                    }
                }
            }
        }
        else
        {
            // do manual load balancing here
            /*const int ithread = omp_get_thread_num();
            sum_array[ithread] = processed_edges;

            #pragma omp barrier
            #pragma omp single
            {
                scan(sum_array, sum_array, static_cast<VNT>(0), total_threads);
            }
            #pragma omp barrier

            VNT local_additive = sum_array[ithread];

            #pragma omp for schedule(static)
            for (VNT i = 0; i < x_nvals; i++)
            {
                VNT ind = x_ids[i];
                ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
                ENT row_end   = _matrix->row_ptr[ind + 1];
                search_array[i] = row_end - row_start + local_additive;
            }

            ENT approx_elems_per_thread = (total_edges - 1) / total_threads + 1;
            ENT expected_tid_left_border = approx_elems_per_thread * ithread;
            ENT expected_tid_right_border = approx_elems_per_thread * (ithread + 1);
            auto low_pos = std::lower_bound(search_array, search_array, expected_tid_left_border);
            auto up_pos = std::lower_bound(search_array, search_array, expected_tid_right_border);

            VNT low_val = low_pos - search_array;
            VNT up_val = min(x_nvals, (VNT)(up_pos - search_array));
            cout << low_val << " vs " << up_val << endl;*/

            //for (VNT i = low_val; i < up_val; i++)
            #pragma omp for schedule(guided, 128)
            for (VNT i = 0; i < x_nvals; i++)
            {
                VNT ind = x_ids[i];
                X x_val = x_vals[i];
                ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
                ENT row_end   = _matrix->row_ptr[ind + 1];
                typename tbb::concurrent_hash_map<VNT, Y>::accessor a;

                for (ENT j = row_start; j < row_end; j++)
                {
                    VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
                    A mat_val = _matrix->vals[j];

                    if(!map_output.find(a, dest_ind)) {
                        map_output.insert(a, dest_ind);
                        a->second = add_op(identity_val, mul_op(mat_val, x_val));
                    } else {
                        a->second = add_op(a->second, mul_op(mat_val, x_val));
                    }
                }
            }
        }
    }

    if(_mask != 0) // apply mask and save results
    {
        Desc_value mask_field;
        _desc->get(GrB_MASK, &mask_field);
        if(!_mask->is_dense())
            std::cout << "warning! costly mask conversion to dense in spmspv seq_mask" << std::endl;
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
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
