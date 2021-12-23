#pragma once

#define GrB_Index Index

#define GrB_ALL NULL

template <typename T>
struct LAGraph_Graph
{
    lablas::Matrix<float> *A;
    lablas::Matrix<float> *AT;
};

#define GrB_TRY( CallInstruction ) { \
    CallInstruction; \
}

#define LG_CLEAR_MSG printf("starting lagraph alg");

enum GrB_Type
{
    GrB_FP32 = 0
};
