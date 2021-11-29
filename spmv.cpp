#include "src/gb_kun.h"

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
        print_omp_stats();
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

        lablas::Vector<float> w(el.vertices_count);
        lablas::Vector<float> u(el.vertices_count);

        w.fill(0.0);
        u.fill(1.0);

        lablas::mxv<float, float, float, float>(&w,NULL, lablas::PlusMonoid<float>(), lablas::PlusMonoid<float>(),&matrix, &u, &desc);

        int num_runs = 100;
        double avg_time = 0;
        for(int run = 0; run < num_runs; run++)
        {
            w.fill(0.0);
            u.fill(1.0);
            double t1 = omp_get_wtime();
            lablas::mxv<float, float, float, float>(&w,NULL, lablas::PlusMonoid<float>(), lablas::PlusMonoid<float>(),&matrix, &u, &desc);
            double t2 = omp_get_wtime();
            avg_time += (t2 - t1) / num_runs;
        }

        double perf = 2.0*el.edges_count/(avg_time*1e9);
        double bw = (3.0*sizeof(float)+sizeof(VNT))*el.edges_count/(avg_time*1e9);
        cout << "SPMV perf: " << perf << " GFlop/s" << endl;
        cout << "SPMV BW: " << bw << " GB/s" << endl;
        save_to_file("./output/perf.txt", perf);
        save_to_file("./output/bw.txt", bw);

        if(parser.check())
        {
            lablas::Matrix<float> check_matrix;
            check_matrix.set_preferred_matrix_format(CSR);
            check_matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);
            lablas::Vector<float> w_check(el.vertices_count);

            u.fill(1.0);
            w_check.fill(0.0);
            lablas::mxv<float, float, float, float>(&w_check,NULL, lablas::PlusMonoid<float>(), lablas::PlusMonoid<float>(),&matrix, &u, &desc);

            if(w == w_check)
            {
                cout << "Vectors are equal" << endl;
            }
            else
            {
                cout << "Vectors are NOT equal" << endl;
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

