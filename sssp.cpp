#include "src/gb_kun.h"

#include "algorithms/sssp/sssp.hpp"
#include "algorithms/sssp/sssp_traditional.hpp"

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

        lablas::Vector<float> distances(size);

        Index source_vertex = 0;

        lablas::Descriptor desc;
        lablas::sssp(&distances, graph.A, source_vertex, &desc);

        if(parser.check())
        {
            lablas::Vector<float> check_distances(size);

            lablas::algorithm::sssp_traditional_dijkstra(&check_distances, &matrix, source_vertex);

            if(distances == check_distances)
            {
                cout << "SSSP distances are equal" << endl;
            }
            else
            {
                cout << "SSSP distances are NOT equal" << endl;
            }
        }

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
