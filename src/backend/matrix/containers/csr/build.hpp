/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::numa_aware_realloc()
{
    if(num_sockets_used() == 1)
        return;
    VNT num_rows = this->nrows;

    ENT *new_row_ptr;
    T *new_vals;
    VNT *new_col_ids;
    VNT *new_row_degrees;

    MemoryAPI::allocate_array(&new_row_ptr, num_rows + 1);
    MemoryAPI::allocate_array(&new_row_degrees, num_rows);
    MemoryAPI::allocate_array(&new_col_ids, this->nnz);
    MemoryAPI::allocate_array(&new_vals, this->nnz);

    auto offsets = get_load_balancing_offsets();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VNT first_row = offsets[tid].first;
        VNT last_row = offsets[tid].second;

        for(VNT row = first_row; row < last_row; row++)
        {
            new_row_ptr[row] = this->row_ptr[row];
            new_row_degrees[row] = this->row_degrees[row];
            for(ENT j = this->row_ptr[row]; j < this->row_ptr[row + 1]; j++)
            {
                new_col_ids[j] = this->col_ids[j];
                new_vals[j] = this->vals[j];
            }
        }
        new_row_ptr[last_row] = this->row_ptr[last_row];
    }

    // free old ones
    MemoryAPI::free_array(this->row_ptr);
    MemoryAPI::free_array(this->col_ids);
    MemoryAPI::free_array(this->vals);
    MemoryAPI::free_array(this->row_degrees);

    // copy new pointers into old
    row_ptr = new_row_ptr;
    vals = new_vals;
    col_ids = new_col_ids;
    row_degrees = new_row_degrees;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::calculate_degrees()
{
    #pragma omp parallel for
    for(VNT row = 0; row < this->nrows; row++)
    {
        row_degrees[row] = row_ptr[row + 1] - row_ptr[row];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _nrows, VNT _ncols, ENT _nnz)
{
    vector<vector<pair<VNT, T>>> tmp_csr;

    edges_list_to_vector_of_vectors(_row_ids, _col_ids, _vals, _nrows, _nnz, tmp_csr);

    resize(_nrows, _ncols, _nnz);

    vector_of_vectors_to_csr(tmp_csr, row_ptr, col_ids, vals);

    calculate_degrees();

    get_load_balancing_offsets();

    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(vector<vector<pair<VNT, T>>> &_tmp_csr, VNT _nrows, VNT _ncols)
{
    resize(_nrows, _ncols, estimate_nnz_in_vector_of_vectors(_tmp_csr));

    vector_of_vectors_to_csr(_tmp_csr, row_ptr, col_ids, vals);

    calculate_degrees();
    get_load_balancing_offsets();
    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build_from_csr_arrays(const ENT *_row_ptrs,
                                         const VNT *_col_ids,
                                         const T *_vals,
                                         VNT _nrows,
                                         VNT _ncols,
                                         ENT _nnz)
{
    resize(_nrows, _ncols, _nnz);

    MemoryAPI::copy(this->row_ptr, _row_ptrs, _nrows + 1);
    MemoryAPI::copy(this->col_ids, _col_ids, _nnz);
    MemoryAPI::copy(this->vals, _vals, _nnz);

    calculate_degrees();
    get_load_balancing_offsets();
    //numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
