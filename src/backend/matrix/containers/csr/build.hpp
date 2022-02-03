/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::prepare_vg_lists(int _target_socket)
{
    // set thresholds
    ENT step = 16;
    ENT first = 4;
    vertex_groups[0].set_thresholds(0, first);
    for(int i = 1; i < (vg_num - 1); i++)
    {
        vertex_groups[i].set_thresholds(first, first*step);
        first *= step;
    }
    vertex_groups[vg_num - 1].set_thresholds(first, INT_MAX);

    // push back vertices to each group using O|V| work
    for(VNT row = 0; row < size; row++)
    {
        ENT connections_count = row_ptr[row + 1] - row_ptr[row];
        for(int vg = 0; vg < vg_num; vg++)
            if(vertex_groups[vg].in_range(connections_count))
                vertex_groups[vg].push_back(row);
    }

    // create optimized representation for each socket
    for(int i = 0; i < vg_num; i++)
    {
        vertex_groups[i].finalize_creation(_target_socket);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::check_if_static_can_be_used()
{
    VNT num_rows = this->size; // fixme

    int cores_num = omp_get_max_threads();
    static_ok_to_use = true;
    #pragma omp parallel
    {
        ENT core_edges = 0;
        #pragma omp for
        for(VNT row = 0; row < num_rows; row++)
        {
            VNT connections = row_ptr[row + 1] - row_ptr[row];
            core_edges += connections;
        }

        double real_percent = 100.0*((double)core_edges/nnz);
        double supposed_percent = 100.0/cores_num;

        if(fabs(real_percent - supposed_percent) > 3) // if difference is more than 5%, static not ok to use
            static_ok_to_use = false;
    }
    cout << "static is ok to use: " << static_ok_to_use << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::numa_aware_realloc()
{
    ENT *new_row_ptr;
    T *new_vals;
    VNT *new_col_ids;

    MemoryAPI::allocate_array(&new_row_ptr, this->size + 1);
    MemoryAPI::allocate_array(&new_col_ids, this->nnz);
    MemoryAPI::allocate_array(&new_vals, this->nnz);

    #pragma omp parallel
    {
        /*#pragma omp for nowait schedule(static, 1)
        for(VNT i = 0; i < this->large_degree_threshold; i++)
        {
            new_row_ptr[i] = this->row_ptr[i];
            //connections_count[i] = this->row_ptr[i + 1] - this->row_ptr[i];

            for(ENT j = this->row_ptr[i]; j < this->row_ptr[i + 1]; j++)
            {
                new_col_ids[j] = this->col_ids[j];
                new_vals[j] = this->vals[j];
            }
        }

        #pragma omp for nowait schedule(static, CSR_SORTED_BALANCING)
        for(VNT i = this->large_degree_threshold; i < this->size; i++)
        {
            new_row_ptr[i] = this->row_ptr[i];

            for(ENT j = this->row_ptr[i]; j < this->row_ptr[i + 1]; j++)
            {
                new_col_ids[j] = this->col_ids[j];
                new_vals[j] = this->vals[j];
            }
        }*/
    }

    // free old ones
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);

    // copy new pointers into old
    row_ptr = new_row_ptr;
    vals = new_vals;
    col_ids = new_col_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nnz, int _target_socket)
{
    vector<vector<pair<VNT, T>>> tmp_csr;

    edges_list_to_vector_of_vectors(_row_ids, _col_ids, _vals, _size, _nnz, tmp_csr);

    resize(_size, _nnz, _target_socket);

    vector_of_vectors_to_csr(tmp_csr, row_ptr, col_ids, vals);

    prepare_vg_lists(_target_socket);

    check_if_static_can_be_used();

    //numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(vector<vector<pair<VNT, T>>> &_tmp_csr, int _target_socket)
{
    resize(_tmp_csr.size(), estimate_nnz_in_vector_of_vectors(_tmp_csr), _target_socket);

    vector_of_vectors_to_csr(_tmp_csr, row_ptr, col_ids, vals);

    prepare_vg_lists(_target_socket);

    check_if_static_can_be_used();

    //numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
