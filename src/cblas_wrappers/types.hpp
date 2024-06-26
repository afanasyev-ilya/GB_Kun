#pragma once

#define GrB_Index Index

#define GrB_ALL NULL

template <typename T>
struct LAGraph_Graph
{
    lablas::Matrix<T> *A;
    lablas::Matrix<T> *AT;

    lablas::Vector<GrB_Index> *rowdegree;
    lablas::Vector<GrB_Index> *coldegree;

    LAGraph_Graph(lablas::Matrix<T> &_matrix)
    {
        Index nrows, ncols;
        _matrix.get_nrows(&nrows);
        _matrix.get_ncols(&ncols);
        A = &_matrix;
        AT = &_matrix;
        rowdegree = new lablas::Vector<Index>(nrows);
        coldegree = new lablas::Vector<Index>(ncols);
        rowdegree->build(_matrix.get_rowdegrees(), nrows);
        coldegree->build(_matrix.get_coldegrees(), ncols);
    }

    ~LAGraph_Graph()
    {
        delete rowdegree;
        delete coldegree;
    }
};

#define GrB_TRY( CallInstruction ) { \
    LA_Info code = CallInstruction;   \
    if(code != GrB_SUCCESS)          \
        printf("GraphBLAS error: %d at call \"" #CallInstruction "\"\n", (int) code); \
		 /*throw "error in GraphBLAS API function, aborting...";*/ \
}

#define LG_CLEAR_MSG printf("starting lagraph alg\n");

enum GrB_Type
{
    GrB_FP32 = 0,
    GrB_BOOL = 1,
    GrB_INT32 = 2,
    GrB_INT64 = 3
};

// binary operations
#define GrB_DIV_FP32 lablas::div<float>()
#define GrB_MAX_FP32 lablas::maximum<float, float, float>()
#define GrB_PLUS_FP32 lablas::plus<float>()
#define GrB_MINUS_FP32 lablas::minus<float>()
#define GrB_ABS_FP32 lablas::abs<float>()
#define GrB_SECOND_INT64 lablas::second<long long int>()
#define GrB_SECOND_INT32 lablas::second<int>()
#define GrB_SECOND_FLT32 lablas::second<float>()


// semirings
#define LAGraph_plus_second_fp32 lablas::PlusSecondSemiring<float>()
#define LAGraph_structural_bool lablas::StructuralBool<bool>()
#define LAGraph_plus_one_int64 lablas::PlusOneSemiring<long long>()

// monoids
#define GrB_PLUS_MONOID_FP32 lablas::PlusMonoid<float>()
#define GrB_PLUS_MONOID_INT32 lablas::PlusMonoid<int>()
#define GrB_PLUS_MONOID_INT64 lablas::PlusMonoid<long long>()


// descriptors

//#define GrB_NULL (&lablas::GrB_NULL)
#define GrB_DESC_RSC (&lablas::GrB_DESC_RSC)
#define GrB_DESC_C (&lablas::GrB_DESC_C)
#define GrB_DESC_S (&lablas::GrB_DESC_S)
#define GrB_DESC_SC (&lablas::GrB_DESC_SC)

// null
#define GrB_NULL ((long int)NULL)