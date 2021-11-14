#include "src/gb_kun.h"

#define NUM_ITERS 3

#define REPORT_STATS( CallInstruction ) { \
    double bw = CallInstruction;          \
    cout << "BW: " << bw << endl;         \
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

        Matrix<float> matrix;
        matrix.set_preferred_format(parser.get_storage_format());
        Vector<float> x(el.vertices_count);
        Vector<float> y(el.vertices_count);

        matrix.build(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

        x.set_constant(1.0);
        y.set_constant(0.0);
        SpMV(matrix, x, y);

        y.set_constant(0.0);
        double t1 = omp_get_wtime();
        SpMV(matrix, x, y);
        double t2 = omp_get_wtime();
        cout << "SPMV perf: " << 2.0*el.edges_count/((t2-t1)*1e9) << " GFlop/s" << endl;
        cout << "BW: " << (3.0*sizeof(float)+sizeof(VNT))*el.edges_count/((t2-t1)*1e9) << " GB/s" << endl;

        if(parser.check())
        {
            Matrix<float> check_matrix;
            matrix.set_preferred_format(CSR);
            Vector<float> z(el.vertices_count);
            check_matrix.build(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

            x.set_constant(1.0);
            z.set_constant(0.0);
            SpMV(check_matrix, x, z);

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
