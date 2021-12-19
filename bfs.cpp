#include "src/gb_kun.h"

#include "algorithms/bfs/bfs.hpp"
#include "algorithms/bfs/bfs_td.hpp"
#include "algorithms/bfs/bfs_traditional.hpp"

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

        VNT source_vertex = rand() % matrix.get_nrows();
        lablas::algorithm::bfs(&levels, &matrix, source_vertex, &desc);

        if(parser.check())
        {
            lablas::Matrix<float> check_matrix;
            LA_Info info = check_matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);
            check_matrix.set_preferred_matrix_format(CSR);
            lablas::Vector<float> check_levels(check_matrix.get_nrows());

            lablas::algorithm::bfs_traditional(&check_levels, &check_matrix, source_vertex);

            if(levels == check_levels)
            {
                cout << "BFS levels are equal" << endl;
            }
            else
            {
                cout << "BFS levels are NOT equal" << endl;
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
