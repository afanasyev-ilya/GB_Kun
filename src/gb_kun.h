#pragma once

#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <omp.h>

using namespace std;

#define VNT int
#define ENT int

#include "helpers/memory_API/memory_API.h"
#include "helpers/random_generator/random_generator.h"
#include "helpers/graph_generation/graph_generation.h"

#include "core/matrix/csr/csr_matrix.h"
#include "core/vector/dense_vector/dense_vector.h"

#include "core/spmv/spmv.h"
