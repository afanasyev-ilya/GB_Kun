#pragma once

#ifdef __USE_TBB__
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/partitioner.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#endif
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <climits>
#include <map>
#include <unordered_map>
#include <sstream>
#include <string>
#include <list>
#include <cassert>
#include <queue>
#include <functional>
#include <set>
#include <unordered_set>
#include <set>
#include <chrono>

#include <omp.h>

typedef long long Index;

using namespace std;

#include "helpers/hashmap/tsl/hopscotch_map.h"
#include "helpers/numa_stats/numa_stats.h"
#include "backend/la_backend.h"
#include "helpers/balancing/balancing.h"
#include "helpers/stats/stats.h"
#include "helpers/memory_API/memory_API.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/cmd_parser/cmd_parser.h"
#include "helpers/graph_generation/graph_generation.h"
#include "helpers/parallel_primitives/primitives.h"
#include "helpers/lib_kernels/format_conversions.h"
#include "helpers/timer/timer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DenseVector;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backend

#include "backend/descriptor/descriptor.h"

#include "backend/vector/vector.h"
#include "backend/matrix/matrix.h"
#include "backend/spmv/spmv.h"
#include "backend/spmspv/spmspv.h"
#include "backend/spmspm/spmspm.h"

#include "backend/operations/operations.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cpp interfaces

#include "cpp_graphblas/cpp_graphblas.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cblas interfaces

#include "cblas_wrappers/cblas_wrappers.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "helpers/lib_kernels/init_matrix.h"
#include "helpers/lib_kernels/select_non_vertex.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

