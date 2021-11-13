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

#define VNT int
#define ENT int

#include "helpers/memory_API/memory_API.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/graph_generation/graph_generation.h"
#include "helpers/cmd_parser/cmd_parser.h"

#include "backend/vector/vector.h"
#include "backend/matrix/matrix.h"

#include "backend/spmv/spmv.h"


