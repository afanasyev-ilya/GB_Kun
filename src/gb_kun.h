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
class Matrix;
template <typename T>
class MatrixCSR;
template <typename T>
class MatrixSegmentedCSR;
template <typename T>
class MatrixLAV;
template <typename T>
class MatrixCOO;
template <typename T>
class MatrixCellSigmaC;
template <typename T>
class DenseVector;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "backend/descriptor/descriptor.h"

#include "backend/vector/vector.h"
#include "backend/matrix/matrix.h"

#include "backend/spmv/spmv.h"
#include "common/matrix.hpp"
#include "common/vector.hpp"
#include "common/descriptor.hpp"
#include "common/operations.hpp"

