#include "src/gb_kun.h"

#include "algorithms/sssp/sssp.hpp"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Matrix<float> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        GrB_Index size;
        matrix.get_nrows(&size);
        lablas::Vector<float> levels(size);

        LAGraph_Graph<float> graph(matrix);

        lablas::Vector<float> v(size);

        lablas::Descriptor desc;
        lablas::sssp(&v, graph.A, 0, &desc, 100);

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
