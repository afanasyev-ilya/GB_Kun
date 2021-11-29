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
        LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);


        lablas::Vector<float> x(el.vertices_count);
        lablas::Vector<float> y(el.vertices_count);
        //matrix.set_preferred_format(parser.get_storage_format());
        //        Vector<float> x(el.vertices_count);
        //        Vector<float> y(el.vertices_count);

        //matrix.build(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);
        x.fill(1.0);
        y.fill(0.0);
        //        x.set_constant(1.0);
        //        y.set_constant(0.0);
        lablas::mxv<float, float, float, float>(&x,NULL, lablas::PlusMonoid<float>(), lablas::PlusMonoid<float>(),&matrix, &y, &desc);
        //        SpMV(matrix, x, y, desc);

        int num_runs = 100;
        double avg_time = 0;
        for(int run = 0; run < num_runs; run++)
        {
            y.fill(0.0);
            double t1 = omp_get_wtime();
            lablas::mxv<float, float, float, float>(&x,NULL, lablas::PlusMonoid<float>(), lablas::PlusMonoid<float>(),&matrix, &y, &desc);
            double t2 = omp_get_wtime();
            avg_time += (t2 - t1) / num_runs;
        }

        double perf = 2.0*el.edges_count/(avg_time*1e9);
        double bw = (3.0*sizeof(float)+sizeof(VNT))*el.edges_count/(avg_time*1e9);
        cout << "SPMV perf: " << perf << " GFlop/s" << endl;
        cout << "SPMV BW: " << bw << " GB/s" << endl;
        save_to_file("./output/perf.txt", perf);
        save_to_file("./output/bw.txt", bw);

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

