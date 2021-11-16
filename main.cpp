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

int main(int argc, char **argv)
{
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

        Descriptor desc(el.vertices_count);

        Matrix<float> matrix;
        matrix.set_preferred_format(parser.get_storage_format());
        Vector<float> x(el.vertices_count);
        Vector<float> y(el.vertices_count);

        matrix.build(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

        x.set_constant(1.0);
        y.set_constant(0.0);
        SpMV(matrix, x, y, desc);

        int num_runs = 100;
        double avg_time = 0;
        for(int run = 0; run < num_runs; run++)
        {
            y.set_constant(0.0);
            double t1 = omp_get_wtime();
            SpMV(matrix, x, y, desc);
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
            Matrix<float> check_matrix;
            matrix.set_preferred_format(CSR);
            Vector<float> z(el.vertices_count);
            check_matrix.build(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

            x.set_constant(1.0);
            z.set_constant(0.0);
            SpMV(check_matrix, x, z, desc);

            if(y == z)
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
