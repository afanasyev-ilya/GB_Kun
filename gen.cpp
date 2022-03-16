#define NEED_GEMM
#include "src/gb_kun.h"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        EdgeListContainer<float> el;

        VNT vertices_count = pow(2.0, parser.get_scale());
        ENT edges_count = vertices_count*parser.get_avg_degree();

        if(parser.get_synthetic_graph_type() == RMAT_GRAPH)
            GraphGenerationAPI::RMAT(el, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
        else if(parser.get_synthetic_graph_type() == RANDOM_UNIFORM_GRAPH)
            GraphGenerationAPI::random_uniform(el, vertices_count, edges_count, DIRECTED_GRAPH);
        else if(parser.get_synthetic_graph_type() == HPCG_GRAPH)
            GraphGenerationAPI::HPCG(el, 128, 128, 128, parser.get_avg_degree());

        el.save_as_mtx(parser.get_out_file_name());
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

