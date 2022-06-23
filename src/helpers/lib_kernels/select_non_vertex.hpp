#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * select_non_trivial_vertex function.
 * @brief returns random non trivial vertex from the range.
 * @param _matrix matrix representing a graph
 * @param _range range in which a vertex is to be selected
*/

template <typename T>
Index select_non_trivial_vertex(lablas::Matrix<T> &_matrix, Index _range = -1)
{
    Index max_val = _range;
    if(_range == -1) // not provided
    {
        max_val = min(_matrix.ncols(), _matrix.nrows());
    }
    else
    {
        max_val = min(_range, min(_matrix.ncols(), _matrix.nrows()));
    }

    Index vertex = 0;
    srand(time(NULL));
    do {
        vertex = rand() %  max_val;
    } while((_matrix.get_rowdegrees()[vertex] == 0) || (_matrix.get_coldegrees()[vertex] == 0));
    return vertex;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
