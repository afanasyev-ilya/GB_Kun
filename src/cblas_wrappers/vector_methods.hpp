#pragma once

template <typename T>
LA_Info GrB_Vector_new(lablas::Vector<T> **_vector, GrB_Type _type, GrB_Index _size)
{
    // types can be checked
    *_vector = new lablas::Vector<T>(_size);
    return GrB_SUCCESS;
}

template <typename T>
LA_Info GrB_free(lablas::Vector<T> **_vector)
{
    // types can be checked
    delete (*_vector);
    *_vector = NULL;
    return GrB_SUCCESS;
}