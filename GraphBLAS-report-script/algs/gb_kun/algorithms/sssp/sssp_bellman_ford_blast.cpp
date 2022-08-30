#include "gb_kun.h"

#include "sssp.hpp"
#include "sssp_blast.hpp"
#include "sssp_traditional.hpp"

#include <stdlib.h>

//see select_non_vertex.hpp at GB_kun project
template <typename T>
Index select_ntv(lablas::Matrix<T> &_matrix, int seed, Index _range = -1)
{
    Index max_val = _range;
    if(_range == -1) // not provided
    {
        max_val = min(_matrix.ncols(), _matrix.nrows());
    }
    else
    {
        max_val = min(_range, min(_matrix.ncols(), _matrix.nrows()));
    }

    Index vertex = 0;
    srand(seed);
    do {
        vertex = rand() %  max_val;
    } while((_matrix.get_rowdegrees()[vertex] == 0) || (_matrix.get_coldegrees()[vertex] == 0));
    return vertex;
}

int main(int argc, char **argv) {
  lablas::Descriptor desc;

  lablas::Matrix<float> matrix;

  auto start = std::chrono::steady_clock::now();
  // matrix.set_preferred_matrix_format(FORMAT);
  matrix.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  GrB_Index size;
  matrix.get_nrows(&size);
  LAGraph_Graph<float> graph(matrix);

  lablas::Vector<float> distances(size);

  Index source_vertex = 0;

  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    source_vertex = select_ntv(matrix, i);
    std::cout << "source : " << source_vertex << ", ineration : " << i << '\n';
    start = std::chrono::steady_clock::now();
    lablas::algorithm::sssp_bellman_ford_blast(&distances, &matrix,
                                               source_vertex, &desc);
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    GrB_Vector_clear(&distances);
  }

  myfile.close();

  return 0;
}