#ifndef GB_KUN_DIMENSIONS_HPP
#define GB_KUN_DIMENSIONS_HPP
#include "vector.hpp"
#include "matrix.hpp"
#include "operations.hpp"
#include <string>
//namespace lablas {
//
//    template <typename U>
//    inline LA_Info checkDimVecNvals(const Vector<U>*   u,
//                                 const std::string& str) {
//        if (u == NULL) return GrB_INVALID_OBJECT;
//        Index u_nvals;
//        u->nvals(&u_nvals);
//        if (u_nvals == 0) {
//            std::cout << str << std::endl;
//            return GrB_INVALID_OBJECT;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename a, typename b>
//    inline LA_Info checkDimRowCol(const Matrix<a>* A,
//                               const Matrix<b>* B,
//                               const std::string& str) {
//        if (A == NULL || B == NULL) return GrB_SUCCESS;
//        Index A_nrows, B_ncols;
//        A->nrows(&A_nrows);
//        B->ncols(&B_ncols);
//        if (A_nrows != B_ncols) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename a, typename b>
//    inline LA_Info checkDimRowRow(const Matrix<a>* A,
//                               const Matrix<b>* B,
//                               const std::string& str) {
//        if (A == NULL || B == NULL) return GrB_SUCCESS;
//        Index A_nrows, B_nrows;
//        A->nrows(&A_nrows);
//        B->nrows(&B_nrows);
//        if (A_nrows != B_nrows) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename a, typename b>
//    inline LA_Info checkDimColCol(const Matrix<a>* A,
//                               const Matrix<b>* B,
//                               const std::string& str) {
//        if (A == NULL || B == NULL) return GrB_SUCCESS;
//        Index A_ncols, B_ncols;
//        A->ncols(&A_ncols);
//        B->ncols(&B_ncols);
//        if (A_ncols != B_ncols) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename a, typename U>
//    inline LA_Info checkDimRowSize(const Matrix<a>*   A,
//                                const Vector<U>*   u,
//                                const std::string& str) {
//        if (A == NULL || u == NULL) return GrB_SUCCESS;
//        Index A_nrows, u_size;
//        A->nrows(&A_nrows);
//        u->size(&u_size);
//        if (A_nrows != u_size) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename a, typename U>
//    inline LA_Info checkDimColSize(const Matrix<a>* A,
//                                const Vector<U>* u,
//                                const std::string& str) {
//        if (A == NULL || u == NULL) return GrB_SUCCESS;
//        Index A_ncols, u_size;
//        A->ncols(&A_ncols);
//        u->size(&u_size);
//        if (A_ncols != u_size) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//
//    template <typename U, typename W>
//    inline LA_Info checkDimSizeSize(const Vector<U>*   u,
//                                 const Vector<W>*   w,
//                                 const std::string& str) {
//        if (u == NULL || w == NULL) return GrB_SUCCESS;
//        Index u_size, w_size;
//        u->
//        w->size(&w_size);
//        if (u_size != w_size) {
//            std::cout << str << std::endl;
//            return GrB_DIMENSION_MISMATCH;
//        }
//        return GrB_SUCCESS;
//    }
//}  // namespace graphblas

#endif //GB_KUN_DIMENSIONS_HPP
