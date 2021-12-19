#include "src/gb_kun.h"

template<typename T>
void test_locality(int argc, char **argv)
{
    //print_omp_stats();
    Parser parser;
    parser.parse_args(argc, argv);

    EdgeListContainer<T> el;
    GraphGenerationAPI::generate_synthetic_graph(el, parser);

    lablas::Descriptor desc;

    lablas::Matrix<T> matrix;

    const std::vector<VNT> src_ids(el.src_ids);
    const std::vector<VNT> dst_ids(el.dst_ids);
    std::vector<T> edge_vals(el.edge_vals);
    matrix.set_preferred_matrix_format(CSR);
    LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);

    lablas::Vector<T> w(el.vertices_count);
    lablas::Vector<T> u(el.vertices_count);

    int num_runs = 10;
    double avg_time = 0;
    for(int run = 0; run < num_runs; run++)
    {
        w.fill(0.0);
        u.fill(1.0);
        double t1 = omp_get_wtime();
        lablas::mxv<T, T, T, T>(&w, NULL, nullptr, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);
        double t2 = omp_get_wtime();
        avg_time += (t2 - t1) / num_runs;
    }

    double perf = 2.0*matrix.get_nnz()/(avg_time*1e9);
    double bw = (2.0*sizeof(T)+sizeof(Index))*matrix.get_nnz()/(avg_time*1e9);
    cout << "SPMV time: " << avg_time*1000 << " ms" << endl;
    cout << "SPMV perf: " << perf << " GFlop/s" << endl;
    cout << "SPMV BW: " << bw << " GB/s" << endl;
}

int main(int argc, char **argv) {
    try
    {
        test_locality<float>(argc, argv);
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

