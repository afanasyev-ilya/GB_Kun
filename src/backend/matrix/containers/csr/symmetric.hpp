#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCSR<T>::is_symmetric()
{
    std::vector<std::set<VNT>> col_data(ncols);
    std::vector<omp_lock_t> locks(ncols);

    #pragma omp parallel
    {
        #pragma omp for
        for(VNT col = 0; col < ncols; col++)
            omp_init_lock(&locks[col]);

        #pragma omp for
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
