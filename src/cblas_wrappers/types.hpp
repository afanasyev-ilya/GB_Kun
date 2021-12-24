#pragma once

#define GrB_Index Index

#define GrB_ALL NULL
#define NULL_TYPE long int

template <typename T>
struct LAGraph_Graph
{
    lablas::Matrix<float> *A;
    lablas::Matrix<float> *AT;

    lablas::Vector<GrB_Index> *rowdegree;
    lablas::Vector<GrB_Index> *coldegree;
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
    GrB_FP32 = 0
};

// binary operations
#define GrB_DIV_FP32 lablas::div<float>()
#define GrB_MAX_FP32 lablas::maximum<float, float, float>()
#define GrB_PLUS_FP32 lablas::plus<float>()
#define GrB_MINUS_FP32 lablas::minus<float>()
#define GrB_ABS_FP32 lablas::abs<float>()

// semirings
#define LAGraph_plus_second_fp32 lablas::PlusSecondSemiring<float>()