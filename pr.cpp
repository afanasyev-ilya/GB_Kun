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
        SAVE_TIME((init_matrix(matrix, parser)), "read_and_init_matrix_(preprocessing)");

        GrB_Index size;
        matrix.get_nrows(&size);
        lablas::Vector<float> levels(size);

        LAGraph_Graph<float> graph(matrix);
        int max_iter = max(100, parser.get_iterations());

        if(parser.get_algo_name() == "lagraph")
        {
            int iters_taken = 0;
            lablas::Vector<float> *centrality;

            SAVE_TEPS(LAGraph_VertexCentrality_PageRankGAP(&centrality, &graph, &iters_taken, max_iter),
                      "Page_Rank", iters_taken, (graph.AT));
            delete centrality;
        }
        else if(parser.get_algo_name() == "blast")
        {
            lablas::Vector<float> ranks(size);
            lablas::algorithm::page_rank_graph_blast(&ranks, &matrix,  0.85, &desc, parser.get_iterations());
        }
        else
        {
            cout << "Unknown algorithm name in PR" << endl;
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
