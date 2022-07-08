#pragma once

template<typename T>
void Matrix<T>::add_vertex(Index _vertex_id)
{
    if(_format != CSR)
        throw "Error: can not modify non-CSR graph";

    csr_data->add_row(_vertex_id);
    csc_data->add_row(_vertex_id);

    csr_data->apply_modifications();
    csc_data->apply_modifications();
}

template<typename T>
void Matrix<T>::remove_vertex(Index _vertex_id)
{
    if(_format != CSR)
        throw "Error: can not modify non-CSR graph";

    csr_data->remove_row(_vertex_id);
    csc_data->remove_row(_vertex_id);

    csr_data->apply_modifications();
    csc_data->apply_modifications();
}

template<typename T>
void Matrix<T>::add_edge(Index _src_id, Index _dst_id, T _value)
{
    if(_format != CSR)
        throw "Error: can not modify non-CSR graph";

    csr_data->add_val(_src_id, _dst_id, _value);
    csc_data->add_val(_src_id, _dst_id, _value);

    csr_data->apply_modifications();
    csc_data->apply_modifications();
}

template<typename T>
void Matrix<T>::remove_edge(Index _src_id, Index _dst_id)
{
    if(_format != CSR)
        throw "Error: can not modify non-CSR graph";

    csr_data->remove_val(_src_id, _dst_id);
    csc_data->remove_val(_src_id, _dst_id);

    csr_data->apply_modifications();
    csc_data->apply_modifications();
}