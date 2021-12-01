#include "src/gb_kun.h"

#define NUM_ITERS 3

#define REPORT_STATS( CallInstruction ) { \
    double bw = CallInstruction;          \
    cout << "BW: " << bw << endl;         \
}

void save_to_file(const string &_file_name, double _stat)
{
    ofstream stat_file;
    stat_file.open(_file_name, std::ios_base::app);
    stat_file << _stat << endl;
    stat_file.close();
}

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

        cout << "doing BFS..." << endl;

        Index A_nrows;
        matrix.get_nrows(reinterpret_cast<int *>(&A_nrows));

        // Visited vector (use float for now)
        lablas::Vector<float> v(A_nrows);

        VNT s = 1;
        // Frontier vectors (use float for now)
        lablas::Vector<float> f1(A_nrows);
        lablas::Vector<float> f2(A_nrows);

        Desc_value desc_value;
        /* TODO desc_value in descriptor */

        if (desc_value == GrB_PULLONLY) {
            f1.fill(0.f);
            f1.set_element(1.0f, s);
        } else {
            std::vector<Index> indices(1, s);
            std::vector<float> values(1, 1.f);
            f1.build(&indices, &values, 1);
        }

        float iter;
        float succ = 0.f;
        Index unvisited = A_nrows;

        float max_iter = 10;

        for (iter = 1; iter <= max_iter; ++iter) {
            if (desc.get_descriptor()->debug()) {
                std::cout << "=====BFS Iteration " << iter - 1 << "=====\n";

            }

//            if (iter > 1) {
//                std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
//                        "push" : "pull";
//                if (desc->descriptor_.timin/ == 1)
//                    std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
//                    << unvisited << ", " << vxm_mode << ", "
//                    << gpu_tight.ElapsedMillis() << "\n";
//                gpu_tight_time += gpu_tight.ElapsedMillis();
//            }
            unvisited -= static_cast<int>(succ);

            assign<float, float, float, Index>(&v, &f1, NULL, iter, NULL, A_nrows,
                                               &desc);
//            CHECK(desc->toggle(GrB_MASK));
//            vxm<float, float, float, float>(&f2, v, NULL,
//                                            LogicalOrAndSemiring<float>(), &f1, A, desc);

            mxv<float, float, float, float>(&f2, &v, NULL,
                                            NULL, &matrix, &f1, &desc);
//            CHECK(desc->toggle(GrB_MASK));

//            CHECK(f2.swap(&f1));
            reduce<float, float>(&succ, NULL, NULL, &f1, &desc);

            if (desc.get_descriptor()->debug())
                std::cout << "succ: " << succ << std::endl;
            if (succ == 0)
                break;
        }
//        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
//                "push" : "pull";
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
