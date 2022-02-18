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
    for(VNT row = 0; row < nrows; row++)
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
    VNT num_rows = this->nrows;

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
    int cores_num = omp_get_max_threads();
    if(cores_num <= THREADS_PER_SOCKET)
        return;

    VNT num_rows = this->nrows;

    ENT *new_row_ptr;
    T *new_vals;
    VNT *new_col_ids;
    VNT *new_row_degrees;

    MemoryAPI::allocate_array(&new_row_ptr, num_rows + 1);
    MemoryAPI::allocate_array(&new_col_ids, this->nnz);
    MemoryAPI::allocate_array(&new_vals, this->nnz);
    MemoryAPI::allocate_array(&new_row_degrees, num_rows);

    VNT *new_vg_vertices[vg_num];
    for(int vg = 0; vg < vg_num; vg++)
        MemoryAPI::allocate_array(&(new_vg_vertices[vg]), this->vertex_groups[vg].get_size());

    #pragma omp parallel
    {
        if(can_use_static_balancing())
        {
            #pragma omp for schedule(static)
            for(VNT row = 0; row < num_rows; row++)
            {
                new_row_ptr[row] = this->row_ptr[row];
                new_row_degrees[row] = this->row_degrees[row];
                for(ENT j = this->row_ptr[row]; j < this->row_ptr[row + 1]; j++)
                {
                    new_col_ids[j] = this->col_ids[j];
                    new_vals[j] = this->vals[j];
                }
            }
        }
        else
        {
            for(int vg = 0; vg < this->vg_num; vg++)
            {
                const VNT *vertices = this->vertex_groups[vg].get_data();
                VNT *new_vertices = new_vg_vertices[vg];
                VNT vertex_group_size = this->vertex_groups[vg].get_size();

                #pragma omp for nowait schedule(static, CSR_SORTED_BALANCING)
                for(VNT idx = 0; idx < vertex_group_size; idx++)
                {
                    VNT row = vertices[idx];
                    new_vertices[idx] = vertices[idx];
                    new_row_ptr[row] = this->row_ptr[row];
                    new_row_degrees[row] = this->row_degrees[row];

                    for(ENT j = this->row_ptr[row]; j < this->row_ptr[row + 1]; j++)
                    {
                        new_col_ids[j] = this->col_ids[j];
                        new_vals[j] = this->vals[j];
                    }
                }
            }
        }
    }

    if(!can_use_static_balancing())
    {
        for(int vg = 0; vg < vg_num; vg++)
            this->vertex_groups[vg].replace_data(new_vg_vertices[vg]); // it also frees old memory inside !
    }

    // free old ones
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
    MemoryAPI::free_array(row_degrees);

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
void MatrixCSR<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _nrows, VNT _ncols, ENT _nnz,
                         int _target_socket)
{
    vector<vector<pair<VNT, T>>> tmp_csr;

    edges_list_to_vector_of_vectors(_row_ids, _col_ids, _vals, _nrows, _nnz, tmp_csr);

    resize(_nrows, _ncols, _nnz, _target_socket);

    vector_of_vectors_to_csr(tmp_csr, row_ptr, col_ids, vals);

    prepare_vg_lists(_target_socket);

    calculate_degrees();

    check_if_static_can_be_used();

    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::build(vector<vector<pair<VNT, T>>> &_tmp_csr, VNT _nrows, VNT _ncols, int _target_socket)
{
    resize(_nrows, _ncols, estimate_nnz_in_vector_of_vectors(_tmp_csr), _target_socket);

    vector_of_vectors_to_csr(_tmp_csr, row_ptr, col_ids, vals);

    prepare_vg_lists(_target_socket);

    calculate_degrees();

    check_if_static_can_be_used();

    numa_aware_realloc();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
