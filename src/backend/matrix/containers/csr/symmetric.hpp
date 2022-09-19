#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCSR<T>::is_symmetric()
{
    std::vector<std::set<VNT>> col_data(ncols);
    std::vector<omp_lock_t> locks(ncols);

    if(nrows != ncols)
        return false;

    #pragma omp parallel
    {
        #pragma omp for
        for(VNT col = 0; col < ncols; col++)
            omp_init_lock(&locks[col]);

        #pragma omp for schedule(guided, 1024)
        for(VNT row = 0; row < nrows; row++)
        {
            for(ENT i = row_ptr[row]; i < row_ptr[row + 1]; i++)
            {
                VNT col_id = col_ids[i];
                omp_set_lock(&locks[col_id]);
                col_data[col_id].insert(row);
                omp_unset_lock(&locks[col_id]);
            }
        }

        #pragma omp for
        for(VNT col = 0; col < ncols; col++)
            omp_destroy_lock(&locks[col]);

    };

    std::unordered_set<VNT> incorrect_rows;
    VNT incorrect_rows_num = 0;
    #pragma omp parallel for reduction(+: incorrect_rows_num)
    for(VNT row = 0; row < nrows; row++)
    {
        std::set<VNT> cur_row_set;
        for(ENT i = row_ptr[row]; i < row_ptr[row + 1]; i++)
        {
            cur_row_set.insert(col_ids[i]);
        }

        if((cur_row_set.size() != col_data[row].size()) && (cur_row_set != col_data[row]))
        {
            incorrect_rows_num++;
            #ifdef __DEBUG_INFO__
            #pragma omp critical
            {
                incorrect_rows.insert(row);
            }
            #endif
        }
    }

    #ifdef __DEBUG_INFO__
    std::cout << "incorrect_rows_num: " << incorrect_rows_num << endl;
    for(auto it: incorrect_rows)
        std::cout << it << std::endl;
    #endif

    return incorrect_rows_num == 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCSR<T>::is_symmetric_safe()
{
    if(nrows != ncols) { return false; }

    std::set<std::tuple<VNT, VNT, T> > edges;

    for(VNT row = 0; row < nrows; row++) {
        for (ENT cols_idx = row_ptr[row]; cols_idx < row_ptr[row + 1]; ++cols_idx) {
            edges.insert(std::make_tuple(row, col_ids[cols_idx], vals[cols_idx]));
        }
    }

    for (const auto & edge : edges) {
        if (edges.find(std::make_tuple(std::get<1>(edge), std::get<0>(edge), std::get<2>(edge))) == edges.end()) {
            return false;
        }
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::to_symmetric()
{
    std::vector<std::unordered_map<VNT, T>> col_data(ncols);
    std::vector<omp_lock_t> locks(ncols);

    if(nrows != ncols)
    {
        std::cout << "can't made symmetric matrix with unequal number of cols and rows" << std::endl;
        throw "Aborting...";
    }

    #pragma omp parallel
    {
        #pragma omp for
        for(VNT col = 0; col < ncols; col++)
            omp_init_lock(&locks[col]);

        #pragma omp for schedule(guided, 1024)
        for(VNT row_id = 0; row_id < nrows; row_id++)
        {
            for(ENT i = row_ptr[row_id]; i < row_ptr[row_id + 1]; i++)
            {
                VNT col_id = col_ids[i];
                T val = vals[i];

                omp_set_lock(&locks[col_id]);
                col_data[col_id][row_id] = val;
                omp_unset_lock(&locks[col_id]);

            }
        }

        #pragma omp for schedule(guided, 1024)
        for(VNT row_id = 0; row_id < nrows; row_id++)
        {
            for(ENT i = row_ptr[row_id]; i < row_ptr[row_id + 1]; i++)
            {
                VNT col_id = col_ids[i];
                T val = vals[i];
                col_data[row_id][col_id] = val;
            }
        }

        #pragma omp for
        for(VNT col = 0; col < ncols; col++)
            omp_destroy_lock(&locks[col]);
    }

    ENT new_nnz = 0;
    #pragma omp parallel for reduction(+: new_nnz)
    for(VNT row = 0; row < nrows; row++)
    {
        new_nnz += col_data[row].size();
    }

    #ifdef __DEBUG_INFO__
    std::cout << "number of non zero elements increased from " << this->nnz << " to " << new_nnz <<
        " (" << (double)new_nnz/this->nnz << " times)" << std::endl;
    #endif

    this->resize(nrows, ncols, new_nnz);
    vector_of_maps_to_csr(col_data, row_ptr, col_ids, vals);
    calculate_degrees();
    get_load_balancing_offsets();
    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::to_symmetric_safe()
{
    if(nrows != ncols)
    {
        std::cout << "can't made symmetric matrix with unequal number of cols and rows" << std::endl;
        throw "Aborting...";
    }

    std::map<std::pair<VNT, VNT>, T> edges;
    for(VNT row = 0; row < nrows; row++) {
        for (ENT cols_idx = row_ptr[row]; cols_idx < row_ptr[row + 1]; ++cols_idx) {
            if (vals[cols_idx]) {
                edges[std::make_pair(row, col_ids[cols_idx])] = vals[cols_idx];
                edges[std::make_pair(col_ids[cols_idx], row)] = vals[cols_idx];
            }
        }
    }

    VNT new_nrows = nrows;
    VNT new_ncols = ncols;
    VNT new_nnz = edges.size();
    resize(new_nrows, new_ncols, new_nnz);

    std::vector<std::map<VNT, T> > edges_vector_of_map(new_nrows);
    for (const auto & edge_key_value : edges) {
        const auto edge_pair = edge_key_value.first;
        const auto edge_weight = edge_key_value.second;
        edges_vector_of_map[edge_pair.first][edge_pair.second] = edge_weight;
    }

    ENT cur_csr_pos = 0;
    for(VNT i = 0; i < new_nrows; i++) {
        row_ptr[i] = cur_csr_pos;
        row_ptr[i + 1] = row_ptr[i] + edges_vector_of_map[i].size();
        for(const auto &[col_id, val]: edges_vector_of_map[i]) {
            col_ids[cur_csr_pos] = col_id;
            vals[cur_csr_pos] = val;
            ++cur_csr_pos;
        }
    }

    calculate_degrees();
    get_load_balancing_offsets();
    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

