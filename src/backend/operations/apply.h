#pragma once

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "../vector/dense_vector/dense_vector.h"
#include "../vector/sparse_vector/sparse_vector.h"

namespace lablas{
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename U, typename M,
    typename BinaryOpT,     typename UnaryOpT>
LA_Info apply(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              UnaryOpT         op,
              const Vector<U>* u,
              Descriptor*      desc)
{
    Vector<U>* u_t = const_cast<Vector<U>*>(u);

    Storage u_vec_type;
    u->getStorage(&u_vec_type);

    // sparse variant
    if (u_vec_type == GrB_SPARSE) {
        applySparse(w->getSparse(), mask, accum, op, u_t->getSparse(), desc);
        // dense variant
    } else if (u_vec_type == GrB_DENSE) {
        applyDense(w->getDense(), mask, accum, op, u_t->getDense(), desc);
    } else {
        return GrB_UNINITIALIZED_OBJECT;
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename U, typename W, typename M,
        typename BinaryOpT, typename UnaryOpT>
LA_Info applySparse(SparseVector<W>* w,
                    const Vector<M>* mask,
                    BinaryOpT        accum,
                    UnaryOpT         op,
                    SparseVector<U>* u,
                    Descriptor*      desc)
{
    VNT vec_size;
    u->get_nnz(&vec_size);
    std::string accum_type = typeid(accum).name();
    bool use_accum = accum_type.size() > 1;

    for (Index i = 0; i < vec_size; ++i) {
        if (use_accum) {
            w->get_vals()[i] = op(u->get_vals()[i]);
        } else {
            w->get_vals()[i] = op(u->get_vals()[i]);
        }
    }
    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename U, typename W, typename M,
        typename BinaryOpT, typename UnaryOpT>
LA_Info applyDense(DenseVector<W>*  w,
                   const Vector<M>* mask,
                   BinaryOpT        accum,
                   UnaryOpT         op,
                   DenseVector<U>*  u,
                   Descriptor*      desc)
{
    VNT vec_size;
    u->get_size(&vec_size);
    std::string accum_type = typeid(accum).name();
    bool use_accum = accum_type.size() > 1;

    for (Index i = 0; i < vec_size; ++i) {
        if (use_accum) {
            w->get_vals()[i] = op(u->get_vals()[i]);
        } else {
            w->get_vals()[i] = op(u->get_vals()[i]);
        }
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W> *_w,
              const Vector<M> *_mask,
              BinaryOpTAccum _accum,
              BinaryOpT _op,
              const Vector<U> *_u,
              const T _val,
              Descriptor *_desc)
{
    Storage u_vec_type, w_vec_type;
    _u->getStorage(&u_vec_type);
    _w->getStorage(&w_vec_type);

    if(_w->getDense()->get_size() != _u->getDense()->get_size())
        return GrB_DIMENSION_MISMATCH;

    if((u_vec_type == GrB_DENSE) && (w_vec_type == GrB_DENSE))
    {
        Index size = _w->getDense()->get_size();
        W* w_data = _w->getDense()->get_vals();
        const U* u_data = _u->getDense()->get_vals();

        if(_mask != NULL)
        {
            if(_w->getDense()->get_size() != _mask->getDense()->get_size())
                return GrB_DIMENSION_MISMATCH;

            const M *mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for(Index i = 0; i < size; i++)
            {
                if(mask_data[i])
                    w_data[i] = _op(u_data[i], _val);
            }
        }
        else
        {
            #pragma omp parallel for
            for(Index i = 0; i < size; i++)
            {
                w_data[i] = _op(u_data[i], _val);
            }
        }
    }
    else
    {
        throw "Error in apply: non-dense vectors are unsupported";
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpTAccum, typename BinaryOpT>
LA_Info apply(Vector<W> *_w,
              const Vector<M> *_mask,
              BinaryOpTAccum _accum,
              BinaryOpT _op,
              const T _val,
              const Vector<U> *_u,
              Descriptor *_desc)
{
    Storage u_vec_type, w_vec_type;
    _u->getStorage(&u_vec_type);
    _w->getStorage(&w_vec_type);

    if(_w->getDense()->get_size() != _u->getDense()->get_size())
        return GrB_DIMENSION_MISMATCH;

    if((u_vec_type == GrB_DENSE) && (w_vec_type == GrB_DENSE))
    {
        Index size = _w->getDense()->get_size();
        W* w_data = _w->getDense()->get_vals();
        const U* u_data = _u->getDense()->get_vals();

        if(_mask != NULL)
        {
            if(_w->getDense()->get_size() != _mask->getDense()->get_size())
                return GrB_DIMENSION_MISMATCH;

            const M *mask_data = _mask->getDense()->get_vals();
            #pragma omp parallel for
            for(Index i = 0; i < size; i++)
            {
                if(mask_data[i])
                    w_data[i] = _op(_val, u_data[i]);
            }
        }
        else
        {
            #pragma omp parallel for
            for(Index i = 0; i < size; i++)
            {
                w_data[i] = _op(_val, u_data[i]);
            }
        }
    }
    else
    {
        throw "Error in apply: non-dense vectors are unsupported";
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

