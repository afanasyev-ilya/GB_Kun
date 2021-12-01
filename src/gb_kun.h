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

#include <omp.h>

typedef int Index;

using namespace std;

#include "backend/la_backend.h"
#include "helpers/memory_API/memory_API.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/graph_generation/graph_generation.h"
#include "helpers/cmd_parser/cmd_parser.h"
#include "helpers/parallel_primitives/primitives.h"
#include "helpers/stats/stats.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DenseVector;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "backend/descriptor/descriptor.h"

#include "backend/vector/vector.h"
#include "backend/matrix/matrix.h"
#include "backend/spmv/spmv.h"

#include "cpp_graphblas/matrix.hpp"
#include "cpp_graphblas/vector.hpp"
#include "cpp_graphblas/descriptor.hpp"
#include "cpp_graphblas/operations.hpp"

