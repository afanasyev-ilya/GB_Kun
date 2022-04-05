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
        graph.A->get_matrix()->sort_csc_rows("STL_SORT");
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
        if (parser.check()) {
            uint64_t slow_ntriangles = 0;
            lablas::Matrix<int> C;
            lablas::Matrix<int> D;
            lablas::Matrix<int>* A = graph.A;
            #define MASK_NULL static_cast<const lablas::Matrix<int>*>(NULL)
            lablas::mxm(&C, MASK_NULL, lablas::second<int>(),
                        lablas::PlusMultipliesSemiring<int>(), A, A, &lablas::GrB_DESC_IKJ);
            lablas::mxm(&D, MASK_NULL, lablas::second<int>(),
                        lablas::PlusMultipliesSemiring<int>(), &C, A, &lablas::GrB_DESC_IKJ);
            #undef MASK_NULL
            for (int i = 0; i < D.get_matrix()->get_nrows(); ++i) {
                slow_ntriangles += D.get_matrix()->get_csr()->get(i, i);
            }
            uint64_t slow_ntriangles_undir = slow_ntriangles / 6;
            uint64_t slow_ntriangles_dir = slow_ntriangles / 3;
            std::cout << "Found triangles on undirected graph with slow algorithm: " << slow_ntriangles_undir << std::endl;
            std::cout << "Found triangles on directed graph with slow algorithm: " << slow_ntriangles_dir << std::endl;
            std::cout << "Error count: "
                      << min((slow_ntriangles_undir > ntriangles ? slow_ntriangles_undir - ntriangles : ntriangles - slow_ntriangles_undir),
                             (slow_ntriangles_dir > ntriangles ? slow_ntriangles_dir - ntriangles : ntriangles - slow_ntriangles_dir))
                      << std::endl;
        }
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
