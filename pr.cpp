#include "src/gb_kun.h"

#include "algorithms/pr/pr.hpp"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Descriptor desc;

        lablas::Matrix<float> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        GrB_Index size;
        matrix.get_nrows(&size);
        lablas::Vector<float> levels(size);

        LAGraph_Graph<float> graph(matrix);

        int iters_taken = 0;
        lablas::Vector<float>* centrality;

        int max_iter = max(100, parser.get_iterations());

        /*SAVE_TEPS(LAGraph_VertexCentrality_PageRankGAP(&centrality, &graph, &iters_taken, max_iter),
                  "Page_Rank", iters_taken, (graph.AT));*/

        lablas::Vector<float> ranks(size);
        lablas::algorithm::page_rank_graph_blast(&ranks, &matrix,  0.85, &desc, parser.get_iterations());

        delete centrality;
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
