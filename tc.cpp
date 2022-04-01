#include "src/gb_kun.h"

#include "algorithms/tc/tc.hpp"


int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();
        lablas::Descriptor desc;

        lablas::Matrix<int> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);
        //matrix.get_matrix()->print();

        Index nrows;
        matrix.get_nrows(&nrows);

        LAGraph_Graph<int> graph(matrix);
        //graph.A->get_matrix()->print();

        //lablas::backend::Vector<int>* label_vector = new lablas::backend::Vector<int>(graph.A->get_matrix()->get_nrows());

        //graph.A->get_matrix()->print_graphviz("./result.dot", lablas::backend::VISUALISE_AS_UNDIRECTED, label_vector);

        uint64_t ntriangles;

        SAVE_TEPS(lablas::algorithm::LAGr_TriangleCount(&ntriangles, &graph,
                                     lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Burkhardt,
                                     lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                     NULL),
                  "TriangleCount", 1,(graph.AT));
        cout << "Found triangles: " << ntriangles << endl;
    }
    catch (string& error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}
