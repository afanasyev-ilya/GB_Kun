/// @file operations.hpp
/// @author Lastname:Firstname
/// @version Revision 1.1
/// @brief CBLAS wrappers operations
/// @details Implements wrappers for implemented base operations for CBLAS interfaces
/// @date June 8, 2022

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const T _val,
                  const lablas::Vector<U>* _u,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _val, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(val, u[i]) */
template <typename W, typename M, typename U, typename BinaryOpTAccum, typename UnaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  UnaryOpT _op,
                  const lablas::Vector<U>* _u,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], val) */
template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_apply(lablas::Vector<W>* _w,
                  const lablas::Vector<M>* _mask,
                  BinaryOpTAccum _accum,
                  BinaryOpT _op,
                  const lablas::Vector<U>* _u,
                  const T _val,
                  lablas::Descriptor* _desc)
{
    return lablas::apply(_w, _mask, _accum, _op, _u, _val, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ value */
template <typename W, typename M, typename U,
        typename BinaryOpTAccum>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpTAccum        _accum,
                   U _value,
                   const GrB_Index *_indices,
                   const GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{
    return lablas::assign(_w, _mask, _accum, _value, _indices, _nindices, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[indexes[i]] = mask[indexes[i]] ^ _u[indexes[i]] */
template <typename W, typename M, typename U, typename BinaryOpTAccum>
LA_Info GrB_assign(lablas::Vector<W>*       _w,
                   const lablas::Vector<M>* _mask,
                   BinaryOpTAccum        _accum,
                   lablas::Vector<U>* _u,
                   const GrB_Index *_indices,
                   const GrB_Index _nindices,
                   lablas::Descriptor*  _desc)
{
    return lablas::assign(_w, _mask, _accum, _u, _indices, _nindices, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]); w is UNION of u an v */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_eWiseAdd(lablas::Vector<W>* _w,
                     const lablas::Vector<M>* _mask,
                     BinaryOpTAccum _accum,
                     BinaryOpT _op,
                     const lablas::Vector<U>* _u,
                     const lablas::Vector<V>* _v,
                     lablas::Descriptor* _desc)
{
    return lablas::eWiseAdd(_w, _mask, _accum, _op, _u, _v, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* w[i] = mask[i] ^ op(u[i], v[i]); w is INTERSECTION of u an v */
template <typename W, typename M, typename U, typename V, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info GrB_eWiseMult(lablas::Vector<W>* _w,
                      const lablas::Vector<M>* _mask,
                      BinaryOpTAccum _accum,
                      BinaryOpT _op,
                      const lablas::Vector<U>* _u,
                      const lablas::Vector<V>* _v,
                      lablas::Descriptor* _desc)
{
    return lablas::eWiseMult(_w, _mask, _accum, _op, _u, _v, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_vxm (lablas::Vector<W>*       _w,
                 const lablas::Vector<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Vector<U>* _u,
                 const lablas::Matrix<a>* _matrix,
                 lablas::Descriptor*      _desc)
{
    return lablas::vxm(_w, _mask, _accum, _op, _u, _matrix, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS MxM Wrapper
///
/// A CBLAS wrapper for MxM algorithm. The selection of mxm algorithm is defined by the _mask parameter and the
/// passed descriptor. This function stores the result of multiplication of two input matrices _a and _b in matrix by
/// pointer _c. Masked multiplication is done if the _mask parameter is not null pointer. Binary operation accumulator
/// and semiring operations are supported as well with parameters _accum and _op.
/// @param[out] _c Pointer to the (empty) matrix object that will contain the result matrix.
/// @param[in] _mask Pointer to the mask matrix
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Semiring operation
/// @param[in] _a Pointer to the first input matrix
/// @param[in] _b Pointer to the second input matrix
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_mxm (lablas::Matrix<W>*       _c,
                 const lablas::Matrix<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Matrix<U>* _a,
                 const lablas::Matrix<a>* _b,
                 lablas::Descriptor*      _desc)
{
    return lablas::mxm(_c, _mask, _accum, _op, _a, _b, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename a, typename U,
        typename BinaryOpTAccum, typename SemiringT>
LA_Info GrB_mxv (lablas::Vector<W>*       _w,
                 const lablas::Vector<M>* _mask,
                 BinaryOpTAccum        _accum,
                 SemiringT        _op,
                 const lablas::Matrix<a>* _matrix,
                 const lablas::Vector<U>* _u,
                 lablas::Descriptor*      _desc)
{
    return lablas::mxv(_w, _mask, _accum, _op, _matrix, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Reduce Wrapper for Vector
///
/// Implements a wrapper over a reduce operation for vector which is basically does
/// w = op(w, u[i]) for each i. Accumulator also could be used as well as descriptor.
/// @param[out] _val Pointer to result value
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Monoid operation
/// @param[in] _u Pointer to the Vector object
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info GrB_reduce(T *_val,
                   BinaryOpTAccum _accum,
                   MonoidT _op,
                   const lablas::Vector<U>* _u,
                   lablas::Descriptor* _desc)
{
    return lablas::reduce(_val, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Reduce Wrapper for Matrix
///
/// Implements a wrapper over a reduce operation for matrix which is basically does
/// w = op(w, u[i, j]) for each i, j
/// @param[out] _val Pointer to result value
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Monoid operation
/// @param[in] _u Pointer to the Matrix object
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info GrB_reduce(T *_val,
                   BinaryOpTAccum _accum,
                   MonoidT _op,
                   const lablas::Matrix<U>* _u,
                   lablas::Descriptor* _desc)
{
    return lablas::reduce(_val, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Normalize Wrapper for Vector
///
/// Implements a wrapper over a reduce operation for vector which is basically does
/// w = op(w, u[i]) for each i. Accumulator also could be used as well as descriptor.
/// @param[out] _val Pointer to squared reduce value
/// @param[in] _accum Binary operation accumulator
/// @param[in] _op Monoid operation
/// @param[in] _u Pointer to the Vector object
/// @param[in] _desc Pointer to the descriptor
/// @result LA_Info status
template <typename T, typename U, typename BinaryOpTAccum, typename MonoidT>
LA_Info GrB_normalize(T *_val,
                   BinaryOpTAccum _accum,
                   MonoidT _op,
                   lablas::Vector<U>* _u,
                   lablas::Descriptor* _desc)
{
    return lablas::normalize(_val, _accum, _op, _u, _desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
LA_Info GrB_Vector_clear(lablas::Vector<T>* _vec)
{
    return _vec->clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Select Wrapper for Vector
///
/// Implements a wrapper over a Select operation for Vector which is basically does apply a select operator
/// (an index unary operator) to the elements of a vector u
/// and accumutale/store the result in vector w. Mask can also be provided.
/// w[i] = accum(w[i], op(u[i], i, 0, val))
/// @param[out] w Pointer to result vector
/// @param[in] mask Pointer to mask vector
/// @param[in] accum Binary operation accumulator
/// @param[in] op Select operation
/// @param[in] u Pointer to the result Vector object
/// @param[in] val Val parameter for Accum
/// @param[in] desc Pointer to the descriptor
/// @result LA_Info status
template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
LA_Info GrB_select(lablas::Vector<W> *w,
                   const lablas::Vector<M> *mask,
                   BinaryOpT accum,
                   SelectOpT op,
                   const lablas::Vector<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, accum, op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Select Wrapper for Vector with default accumulator
///
/// Implements a wrapper over a Select operation for Vector which is basically does apply a select operator
/// (an index unary operator) to the elements of a vector u
/// and accumutale/store the result in vector w. Mask can also be provided.
/// w[i] = accum(w[i], op(u[i], i, 0, val)). Default accumulator is lablas::second<W, T>.
/// @param[out] w Pointer to result vector
/// @param[in] mask Pointer to mask vector
/// @param[in] accum NULL_TYPE accumulator
/// @param[in] op Select operation
/// @param[in] u Pointer to the input Vector object
/// @param[in] val Val parameter for Accum
/// @param[in] desc Pointer to the descriptor
/// @result LA_Info status
template <typename W, typename M, typename U, typename T, typename SelectOpT>
LA_Info GrB_select(lablas::Vector<W> *w,
                   const lablas::Vector<M> *mask,
                   NULL_TYPE accum,
                   SelectOpT op,
                   const lablas::Vector<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, lablas::second<W, T>(), op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief CBLAS Select Wrapper for Matrix with default accumulator
///
/// Implements a wrapper over a Select operation for Matrix which is basically does apply a select operator
/// (an index unary operator) to the elements of a matrix u
/// and accumutale/store the result in vector w. Mask can also be provided.
/// w[i, j] = accum(w[i, j], op(u[i, j], i, j, 0, val))
/// @param[out] w Pointer to result Matrix object
/// @param[in] mask Pointer to mask Matrix
/// @param[in] accum NULL_TYPE accumulator
/// @param[in] op Select operation
/// @param[in] u Pointer to the input Matrix object
/// @param[in] val Val parameter for Accum
/// @param[in] desc Pointer to the descriptor
/// @result LA_Info status
template <typename W, typename M, typename U, typename T, typename SelectOpT>
LA_Info GrB_select(lablas::Matrix<W> *w,
                   const lablas::Matrix<M> *mask,
                   NULL_TYPE accum,
                   SelectOpT op,
                   const lablas::Matrix<U> *u,
                   const T val,
                   lablas::Descriptor *desc)
{
    return lablas::select(w, mask, lablas::second<W, T>(), op, u, val, desc);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
