#include "src/gb_kun.h"

#include "algorithms/bfs.hpp"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        EdgeListContainer<float> el;
        if(parser.get_synthetic_graph_type() == RANDOM_UNIFORM)
        {
            GraphGenerationAPI::random_uniform(el,
                                               pow(2.0, scale),
                                               avg_deg * pow(2.0, scale));
            cout << "Using UNIFORM graph" << endl;
        }
        else if(parser.get_synthetic_graph_type() == RMAT)
        {
            GraphGenerationAPI::R_MAT(el, pow(2.0, scale), avg_deg * pow(2.0, scale), 57, 19, 19, 5);
            cout << "Using RMAT graph" << endl;
        }

        lablas::Descriptor desc;

        lablas::Matrix<float> matrix;
        /* TODO clearance of ELC vectors in order to free storage */
        const std::vector<VNT> src_ids(el.src_ids);
        const std::vector<VNT> dst_ids(el.dst_ids);
        std::vector<float> edge_vals(el.edge_vals);

        matrix.set_preferred_matrix_format(parser.get_storage_format());
        LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);
        lablas::Vector<float> levels(el.vertices_count);

        w.fill(0.0);
        u.fill(1.0);

        // TODO BFS
        cout << "doing BFS..." << endl;

        VNT source_vertex = 0;
        bfs(&levels, &matrix, source_vertex, &desc);
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
