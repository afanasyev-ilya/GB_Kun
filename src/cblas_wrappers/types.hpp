#pragma once

#define GrB_Index Index

#define GrB_ALL NULL

template <typename T>
struct LAGraph_Graph
{
    lablas::Matrix<float> *A;
    lablas::Matrix<float> *AT;

    lablas::Vector<GrB_Index> *rowdegree;
    lablas::Vector<GrB_Index> *coldegree;
};

#define GrB_TRY( CallInstruction ) { \
    CallInstruction; \
}

#define LG_CLEAR_MSG printf("starting lagraph alg\n");

enum GrB_Type
{
    GrB_FP32 = 0
};

// operations

#define GrB_DIV_FP32 lablas::div<float>()