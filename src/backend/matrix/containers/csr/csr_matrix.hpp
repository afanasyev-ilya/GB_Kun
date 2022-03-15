/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::MatrixCSR()
{
    target_socket = 0;
    load_balancing_offsets_set = false;
    alloc(1, 1, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::~MatrixCSR()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::alloc(VNT _nrows, VNT _ncols, ENT _nnz)
{
    this->nrows = _nrows;
    this->ncols = _ncols;
    this->nnz = _nnz;
    target_socket = 0;

    MemoryAPI::allocate_array(&row_ptr, this->nrows + 1);
    MemoryAPI::allocate_array(&col_ids, this->nnz);
    MemoryAPI::allocate_array(&vals, this->nnz);
    MemoryAPI::allocate_array(&row_degrees, this->nrows);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::free()
{
    MemoryAPI::free_array(row_ptr);
    MemoryAPI::free_array(col_ids);
    MemoryAPI::free_array(vals);
    MemoryAPI::free_array(row_degrees);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::resize(VNT _nrows, VNT _ncols, ENT _nnz)
{
    this->free();
    this->alloc(_nrows, _ncols, _nnz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::deep_copy(MatrixCSR<T> *_copy, int _target_socket)
{
    this->free();
    this->nrows = _copy->nrows;
    this->ncols = _copy->ncols;
    this->nnz = _copy->nnz;

    if(_target_socket == -1)
    {
        this->target_socket = _copy->target_socket;
        MemoryAPI::allocate_array(&this->row_ptr, this->nrows + 1);
        MemoryAPI::allocate_array(&this->col_ids, this->nnz);
        MemoryAPI::allocate_array(&this->vals, this->nnz);
        MemoryAPI::allocate_array(&this->row_degrees, this->nrows);
    }
    else
    {
        this->target_socket = _target_socket;
        MemoryAPI::numa_aware_alloc(&this->row_ptr, this->nrows + 1, this->target_socket);
        MemoryAPI::numa_aware_alloc(&this->col_ids, this->nnz, this->target_socket);
        MemoryAPI::numa_aware_alloc(&this->vals, this->nnz, this->target_socket);
        MemoryAPI::numa_aware_alloc(&this->row_degrees, this->nrows, this->target_socket);
    }

    MemoryAPI::copy(this->row_ptr, _copy->row_ptr, _copy->nrows + 1);
    MemoryAPI::copy(this->vals, _copy->vals, _copy->nnz);
    MemoryAPI::copy(this->col_ids, _copy->col_ids, _copy->nnz);
    MemoryAPI::copy(this->row_degrees, _copy->row_degrees, _copy->nrows);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool MatrixCSR<T>::is_non_zero(VNT _row, VNT _col)
{
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(col_ids[i] == _col)
            return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T MatrixCSR<T>::get(VNT _row, VNT _col) const
{
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(col_ids[i] == _col)
            return vals[i];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::print() const
{
    cout << "--------------------\n";
    for(VNT row = 0; row < nrows; row++)
    {
        for(VNT col = 0; col < ncols; col++)
        {
            cout << get(row, col) << " ";
        }
        cout << endl;
    }
    cout << "--------------------\n";

    #ifdef __CSR_PRINT_DATA_ARRAYS__
    cout << "nrows: " << get_num_rows() << endl;
    cout << "ncols: " << get_num_cols() << endl;
    cout << "nnz: " << nnz << endl;

    cout << "row_ptr: [ ";
    for (int i = 0; i < nrows + 1; i++)
    {
        cout << row_ptr[i] << " ";
    }
    cout << "]\n";

    cout << "col_ids: [ ";
    for (int i = 0; i < nrows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            std::cout << col_ids[j] << " ";
        }
        if (i + 1 != nrows) {
            std::cout << "| ";
        }
    }
    cout << "]\n";

    cout << "vals: [ ";
    for (int i = 0; i < nrows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            std::cout << vals[j] << " ";
        }
        if (i + 1 != nrows) {
            std::cout << "| ";
        }
    }
    cout << "]\n";

    cout << "--------------------\n";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
const vector<pair<VNT, VNT>> & MatrixCSR<T>::get_load_balancing_offsets() const
{
    if(!load_balancing_offsets_set) // recalculate then
    {
        vector<ENT> vector_row_ptr;
        vector_row_ptr.assign(this->row_ptr, this->row_ptr + this->nrows + 1);
        balance_matrix_rows(vector_row_ptr, load_balancing_offsets);
        load_balancing_offsets_set = true;
        #ifdef __DEBUG_INFO__
        cout << "CSR load balancing offsets are recalculated!" << endl;
        #endif

    }
    return load_balancing_offsets;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

