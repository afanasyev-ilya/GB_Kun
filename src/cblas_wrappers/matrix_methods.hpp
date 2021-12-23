#pragma once

/*void GrB_Vector_new (&t, GrB_FP32, n)
{

}*/

template<typename T>
LA_Info GrB_Matrix_nrows(GrB_Index *_nrows, lablas::Matrix<T> *_matrix)
{
    _matrix->get_nrows(_nrows);
    return GrB_SUCCESS;
}