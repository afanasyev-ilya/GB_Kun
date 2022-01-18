#pragma once

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
#include <sstream>
#include <string>
#include <list>
#include <cassert>

#include <omp.h>

typedef long long Index;

using namespace std;

#include "backend/la_backend.h"
#include "helpers/memory_API/memory_API.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/cmd_parser/cmd_parser.h"
#include "helpers/graph_generation/graph_generation.h"
#include "helpers/parallel_primitives/primitives.h"
#include "helpers/stats/stats.h"

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

#include "backend/operations/operations.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cpp interfaces

#include "cpp_graphblas/cpp_graphblas.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cblas interfaces

#include "cblas_wrappers/cblas_wrappers.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "helpers/apps_api/init_matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

