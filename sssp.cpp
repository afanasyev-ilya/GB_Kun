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

        int num_tests = 2;
        for(int i = 0; i < num_tests; i++)
        {
            source_vertex = rand() % size;
            cout << "starting from source: " << source_vertex << endl;
            /*SAVE_TEPS((lablas::algorithm::sssp_bf_gbkun(&distances, graph.A, source_vertex));,
                      "sssp", 1, &matrix);*/
            lablas::algorithm::sssp_bf_gbkun(&distances, graph.A, source_vertex);
        }

        cout << "check source vertex: " << source_vertex << endl;

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
