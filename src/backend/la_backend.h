#pragma once

#define VNT Index
#define ENT Index
#define CHECK_PRINT_NUM 16 // 2^4 graph is fully printed

#define __USE_SOCKET_OPTIMIZATIONS__

#define MTX_READ_PARTITION_SIZE 1024

#define SPARSE_VECTOR_THRESHOLD 0.15 // 15% is really optimal according to preliminary tests, at least for BFS

#ifdef __USE_NEC_SX_AURORA__
#define LLC_CACHE_SIZE (16*1024*1024)
#define VECTOR_LENGTH 256
#elif __USE_KUNPENG__
#define LLC_CACHE_SIZE (64*1024*1024)
#define VECTOR_LENGTH 4
#else
#define LLC_CACHE_SIZE (64*1024*1024)
#define VECTOR_LENGTH 4
#endif

// different format settings

// CSR format settings
#define CSR_SORTED_BALANCING 256
#define __CSR_SEG_MERGE_SMALL__
#define __PARALLEL_TRANSPOSE__
//#define __CSR_PRINT_DATA_ARRAYS__

// SEG CSR settings
#define SEG_CSR_CACHE_BLOCK_SIZE (512*1024)
#define SEG_CSR_MERGE_BLOCK_SIZE (32*1024)

// LAV settings
#define HUB_VERTICES 131072

// debug settings
#define __DEBUG_BANDWIDTHS__
#define __DEBUG_INFO__
#define __SHORT_VECTOR_PRINT__
