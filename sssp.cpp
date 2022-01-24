#include "src/gb_kun.h"

#include "algorithms/sssp/sssp.hpp"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();


        lablas::Descriptor desc;

        lablas::Matrix<int> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        Index nrows;
        matrix.get_nrows(&nrows);
        Index source_vertex = rand() % nrows;

        LAGraph_Graph<float> graph(matrix);

    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}
