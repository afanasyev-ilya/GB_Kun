#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CSRVertexGroup<T>::CSRVertexGroup()
{
    max_size = 1;
    size = 1;
    total_nnz = 0;
    MemoryAPI::allocate_array(&ids, size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroup<T>::copy(CSRVertexGroup & _other_group)
{
    this->size = _other_group.size;
    this->max_size = _other_group.size;
    this->total_nnz = _other_group.total_nnz;
    this->resize(this->max_size);
    this->min_nnz = _other_group.min_nnz;
    this->max_nnz = _other_group.max_nnz;

    MemoryAPI::copy(this->ids, _other_group.ids, this->size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool CSRVertexGroup<T>::id_in_range(VNT _src_id, VNT _nnz_count)
{
    if ((_nnz_count >= min_nnz) && (_nnz_count < max_nnz))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroup<T>::add_vertex(VNT _src_id)
{
    ids[size] = _src_id;
    size++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
template<typename CopyCond>
void CSRVertexGroup<T>::copy_data_if(CSRVertexGroup<T> & _full_group, CopyCond copy_cond,VNT *_buffer)
{
    this->size = ParallelPrimitives::copy_if_data(copy_cond, _full_group.ids, this->ids, _full_group.size,
                                                  _buffer, _full_group.size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroup<T>::resize(VNT _new_size)
{
    max_size = _new_size;
    size = _new_size;
    MemoryAPI::free_array(ids);
    if (_new_size == 0)
        MemoryAPI::allocate_array(&ids, 1);
    else
        MemoryAPI::allocate_array(&ids, _new_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroup<T>::print()
{
    cout << "vertex group info: ";
    for (VNT i = 0; i < size; i++)
        cout << ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CSRVertexGroup<T>::~CSRVertexGroup()
{
    MemoryAPI::free_array(ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CSRVertexGroup<T>::build(MatrixVectGroupCSR<T> *_matrix, VNT _bottom, VNT _top)
{
    VNT matrix_size = _matrix->size;

    VNT local_group_size = 0;
    ENT local_group_total_nnz = 0;

    min_nnz = _bottom;
    max_nnz = _top;

    for(VNT src_id = 0; src_id < matrix_size; src_id++)
    {
        VNT nnz_count = _matrix->get_nnz_in_row(src_id);
        if((nnz_count >= _bottom) && (nnz_count < _top))
        {
            local_group_total_nnz += nnz_count;
            local_group_size++;
        }
    }

    resize(local_group_size);
    total_nnz = local_group_total_nnz;

    VNT vertex_pos = 0;
    for(VNT src_id = 0; src_id < matrix_size; src_id++)
    {
        VNT nnz_count = _matrix->get_nnz_in_row(src_id);
        if((nnz_count >= _bottom) && (nnz_count < _top))
        {
            this->ids[vertex_pos] = src_id;
            vertex_pos++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

