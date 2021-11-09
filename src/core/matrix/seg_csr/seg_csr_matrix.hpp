/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSegmentedCSR<T>::MatrixSegmentedCSR()
{
    alloc(1, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSegmentedCSR<T>::~MatrixSegmentedCSR()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSegmentedCSR<T>::alloc(VNT _size, ENT _non_zeroes_num)
{
    this->size = _size;
    this->non_zeroes_num = _non_zeroes_num;

    // TODO
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSegmentedCSR<T>::free()
{
    // TODO
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
