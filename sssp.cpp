#include "src/gb_kun.h"

#include "algorithms/sssp/sssp.hpp"
#include "algorithms/sssp/sssp_traditional.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
Index number_of_unvisited_vertices(lablas::Vector<T> &_distances)
{
    Index result = 0;
    for (auto & e : _distances)
    {
        if(e < std::numeric_limits<T>::max())
           result++;
    }
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void print_visited_stats(lablas::Vector<T> &_distances)
{
    Index visited = number_of_unvisited_vertices(_distances);
    Index total = _distances.size();
    std::cout << "number of visited vertices: " << visited << " / " << total << std::endl << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
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
        LAGraph_Graph<float> graph(matrix);

        lablas::Vector<float> distances(size);

        Index source_vertex = 0;

        lablas::Descriptor desc;
        int num_tests = 3;
        for(int i = 0; i < num_tests; i++)
        {
            source_vertex = rand() % size;
            double sssp_time_ms = 0;
            {
                Timer tm("sssp");
                lablas::algorithm::sssp_bellman_ford_blast(&distances, graph.A, source_vertex, &desc);
                sssp_time_ms = tm.get_time_ms();
            }
            save_teps("sssp", sssp_time_ms, matrix.get_nnz(), 1);
            print_visited_stats(distances);
        }

        if(parser.check())
        {
            cout << "check source vertex: " << source_vertex << endl;
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

