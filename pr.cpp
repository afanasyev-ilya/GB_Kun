#include "src/gb_kun.h"

#include "algorithms/pr/pr.hpp"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        EdgeListContainer<float> el;
        GraphGenerationAPI::generate_synthetic_graph(el, parser);

        lablas::Descriptor desc;

        lablas::Matrix<float> matrix;

        /* TODO clearance of ELC vectors in order to free storage */
        const std::vector<VNT> src_ids(el.src_ids);
        const std::vector<VNT> dst_ids(el.dst_ids);
        std::vector<float> edge_vals(el.edge_vals);

        matrix.set_preferred_matrix_format(parser.get_storage_format());
        LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);
        lablas::Vector<float> levels(el.vertices_count);

        LAGraph_Graph<float> graph(matrix);

        int iters_taken = 0;
        lablas::Vector<float>* centrality;
        LAGraph_VertexCentrality_PageRankGAP(&centrality, &graph, &iters_taken);

        centrality->print();
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
