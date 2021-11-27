#ifndef GB_KUN_OPERATIONS_H
#define GB_KUN_OPERATIONS_H


#include "../../common/types.hpp"
#include "../helpers/cmd_parser/parser_options.h"
#include "../la_backend.h"
#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../spmv/spmv.h"
namespace lablas{
namespace backend{

    template <typename W, typename U, typename a, typename M,
            typename BinaryOpT, typename SemiringT>
            LA_Info vxm(Vector<W>*       w,
                     const Vector<M>* mask,
                     BinaryOpT        accum,
                     SemiringT        op,
                     const Vector<U>* u,
                     const Matrix<a>* A,
                     Descriptor*      desc) {

                Storage matrix_storage;
                Storage vector_storage;
                //TODO same for vector
                A->getStorage(&matrix_storage);


                //TODO check vector type
                if (matrix_storage == GrB_SPARSE) {
                    SpMV(A,u,w,desc);
                }
            }
}
}
#endif //GB_KUN_OPERATIONS_H
