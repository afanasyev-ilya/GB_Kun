#pragma once

#define VNT Index
#define ENT Index
#define CHECK_PRINT_NUM 16 // 2^4 graph is fully printed

#define __USE_SOCKET_OPTIMIZATIONS__ // TODO activate
#define VECTOR_LENGTH 32
#define HUB_VERTICES 131072
#define CSR_VERTEX_GROUPS_NUM 6

#define MTX_READ_PARTITION_SIZE 1024

#define THREADS_PER_SOCKET 48

#define SPARSE_VECTOR_THRESHOLD 0.6

#define SEG_CSR_CACHE_BLOCK_SIZE (512*1024)
#define SEG_CSR_MERGE_BLOCK_SIZE (32*1024)



