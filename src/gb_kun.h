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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//template <typename T>
//class Matrix;
//template <typename T>
//class MatrixCSR;
//template <typename T>
//class MatrixSegmentedCSR;
//template <typename T>
//class MatrixLAV;
//template <typename T>
//class MatrixCOO;
//template <typename T>
//class DenseVector;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "src/backend/descriptor/descriptor.h"

#include "src/backend/vector/vector.h"
#include "src/backend/matrix/matrix.h"

#include "src/backend/spmv/spmv.h"
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"
#include "src/common/descriptor.hpp"
#include "src/common/operations.hpp"

