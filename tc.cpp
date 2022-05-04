#include "src/gb_kun.h"

#include "algorithms/tc/tc.hpp"


int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Matrix<int> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        if (!matrix.is_symmetric()) {
            matrix.to_symmetric();
        }

        Index nrows;
        matrix.get_nrows(&nrows);

        LAGraph_Graph<int> graph(matrix);

        uint64_t ntriangles;
        double t1 = omp_get_wtime();
        SAVE_TEPS(lablas::algorithm::LAGr_TriangleCount(&ntriangles, &graph,
                                     lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Burkhardt,
                                     lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                     NULL), "TriangleCount", 1,(graph.AT));
        double t2 = omp_get_wtime();
        std::cout << "TC time : " << t2 - t1 << std::endl;
        std::cout << "sort time: " << GLOBAL_SORT_TIME << std::endl;
        std::cout << "inner mxm time: " << GLOBAL_INNER_MXM_TIME << std::endl;

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
            std::cout << "Found triangles with slow algorithm: " << slow_ntriangles_undir << std::endl;
            cout << "error_count: " << (slow_ntriangles_undir > ntriangles ? slow_ntriangles_undir - ntriangles : ntriangles - slow_ntriangles_undir) << "/"
                 << A->get_matrix()->get_nrows() * A->get_matrix()->get_nrows() << endl;
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
