/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::MatrixCSR()
{
    target_socket = 0;
    load_balancing_offsets_set = false;
    alloc(1, 1, 1, target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixCSR<T>::~MatrixCSR()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::alloc(VNT _nrows, VNT _ncols, ENT _nnz, int _target_socket)
{
    this->nrows = _nrows;
    this->ncols = _ncols;
    this->nnz = _nnz;
    target_socket = _target_socket;

    MemoryAPI::numa_aware_alloc(&row_ptr, this->nrows + 1, _target_socket);
    MemoryAPI::numa_aware_alloc(&col_ids, this->nnz, _target_socket);
    MemoryAPI::numa_aware_alloc(&vals, this->nnz, _target_socket);
    MemoryAPI::numa_aware_alloc(&row_degrees, this->nrows, _target_socket);
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
void MatrixCSR<T>::resize(VNT _nrows, VNT _ncols, ENT _nnz, int _target_socket)
{
    this->free();
    this->alloc(_nrows, _ncols, _nnz, _target_socket);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixCSR<T>::deep_copy(MatrixCSR<T> *_copy, int _target_socket)
{
    if(_target_socket == -1)
        _target_socket = _copy->target_socket;
    this->resize(_copy->nrows, _copy->ncols, _copy->nnz, _target_socket);

    MemoryAPI::copy(this->row_ptr, _copy->row_ptr, _copy->nrows + 1);
    MemoryAPI::copy(this->vals, _copy->vals, _copy->nnz);
    MemoryAPI::copy(this->col_ids, _copy->col_ids, _copy->nnz);
    MemoryAPI::copy(this->row_degrees, _copy->row_degrees, _copy->nrows);

    for(int vg = 0; vg < vg_num; vg++)
    {
        this->vertex_groups[vg].deep_copy(_copy->vertex_groups[vg], _target_socket);
    }
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
        cout << "CSR load balancing offsets are recalculated!" << endl;

    }
    return load_balancing_offsets;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

