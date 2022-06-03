#include "src/gb_kun.h"

#include "algorithms/tc/tc.hpp"


int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();
        string tc_algorithm_string = parser.get_algo_name();

        lablas::algorithm::LAGraph_TriangleCount_Method tc_algorithm;

        if (tc_algorithm_string == "tc_burkhardt") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Burkhardt;
        } else if (tc_algorithm_string == "tc_cohen") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Cohen;
        } else if (tc_algorithm_string == "tc_sandia") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Sandia;
        } else if (tc_algorithm_string == "tc_sandia2") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Sandia2;
        } else if (tc_algorithm_string == "tc_sandiadot") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_SandiaDot;
        } else if (tc_algorithm_string == "tc_sandia2dot") {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_SandiaDot2;
        } else {
            tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Default;
        }

        lablas::Matrix<int> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        if (parser.check() && !matrix.is_symmetric()) {
            matrix.to_symmetric();
        }

        Index nrows;
        matrix.get_nrows(&nrows);

        LAGraph_Graph<int> graph(matrix);

        uint64_t ntriangles;
        double t1 = omp_get_wtime();
        // For now the most fast and stable TC algorithm is LAGraph_TriangleCount_Burkhardt
        SAVE_TEPS(lablas::algorithm::LAGr_TriangleCount(&ntriangles, &graph, tc_algorithm,
                                     lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                     &lablas::GrB_DESC_ESC_MASKED), "TriangleCount", 1,(graph.AT));
        double t2 = omp_get_wtime();
        #ifdef __DEBUG_INFO__
            std::cout << "TC time : " << t2 - t1 << std::endl;
        #endif

        cout << "Found triangles: " << ntriangles << endl;

        if (parser.check()) {
            uint64_t another_algorithm_tc_count;
            lablas::algorithm::LAGraph_TriangleCount_Method another_tc_algorithm;
            if (tc_algorithm == lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Default) {
                another_tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Cohen;
            } else {
                another_tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Default;
            }
            lablas::algorithm::LAGr_TriangleCount(&another_algorithm_tc_count, &graph, another_tc_algorithm,
                                                  lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                                  &lablas::GrB_DESC_IJK_DOUBLE_SORT);
            uint64_t slight_errors = (another_algorithm_tc_count > ntriangles ? another_algorithm_tc_count - ntriangles : ntriangles - another_algorithm_tc_count);

            uint64_t strong_errors = 0;
            if (graph.A->get_matrix()->get_nrows() < 100) {
                uint64_t slow_ntriangles = 0;
                lablas::Matrix<int> C;
                lablas::Matrix<int> D;
                lablas::Matrix<int> *A = graph.A;
                #define MASK_NULL static_cast<const lablas::Matrix<int>*>(NULL)
                lablas::mxm(&C, MASK_NULL, lablas::second<int>(),
                            lablas::PlusMultipliesSemiring<int>(), A, A, &lablas::GrB_DESC_IKJ);
                lablas::mxm(&D, MASK_NULL, lablas::second<int>(),
                            lablas::PlusMultipliesSemiring<int>(), &C, A, &lablas::GrB_DESC_IKJ);
                #undef MASK_NULL
                for (int i = 0; i < D.get_matrix()->get_nrows(); ++i) {
                    slow_ntriangles += D.get_matrix()->get_csr()->get(i, i);
                }
                uint64_t slow_tc_result = slow_ntriangles / 6;
                #ifdef __DEBUG_INFO__
                    std::cout << "Found triangles with slow algorithm: " << slow_tc_result << std::endl;
                #endif
                strong_errors = (slow_tc_result > ntriangles ? slow_tc_result - ntriangles : ntriangles - slow_tc_result);
            }
            uint64_t errors_cnt = strong_errors + slight_errors;
            cout << "error_count: " << errors_cnt << "/" << graph.A->get_matrix()->get_nrows() * graph.A->get_matrix()->get_nrows() << endl;
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
