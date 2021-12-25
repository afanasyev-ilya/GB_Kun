#include "src/gb_kun.h"

void save_to_file(const string &_file_name, double _stat)
{
    ofstream stat_file;
    stat_file.open(_file_name, std::ios_base::app);
    stat_file << _stat << endl;
    stat_file.close();
}

void report_num_threads(int level)
{
    #pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
               level, omp_get_num_threads());
    }
}

template<typename T>
void test_spmv(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    EdgeListContainer<T> el;
    GraphGenerationAPI::generate_synthetic_graph(el, parser);

    lablas::Descriptor desc;

    lablas::Matrix<T> matrix;
    /* TODO clearance of ELC vectors in order to free storage */
    const std::vector<VNT> src_ids(el.src_ids);
    const std::vector<VNT> dst_ids(el.dst_ids);
    std::vector<T> edge_vals(el.edge_vals);
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);

    lablas::Vector<T> w(el.vertices_count);
    lablas::Vector<T> u(el.vertices_count);

    w.fill(0.0);
    u.fill(1.0);

    #define MASK_NULL static_cast<const lablas::Vector<T>*>(NULL)

    GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);

    int num_runs = 10;
    double avg_time = 0;
    for(int run = 0; run < num_runs; run++)
    {
        // lablas::PlusMultipliesSemiring<T>()
        w.fill(0.0);
        u.fill(1.0);
        double t1 = omp_get_wtime();
        GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);
        double t2 = omp_get_wtime();
        avg_time += (t2 - t1) / num_runs;
    }

    double perf = 2.0*matrix.get_nnz()/(avg_time*1e9);
    double bw = (2.0*sizeof(T)+sizeof(Index))*matrix.get_nnz()/(avg_time*1e9);
    cout << "SPMV time: " << avg_time*1000 << " ms" << endl;
    cout << "SPMV perf: " << perf << " GFlop/s" << endl;
    cout << "SPMV BW: " << bw << " GB/s" << endl;
    save_to_file("./output/perf.txt", perf);
    save_to_file("./output/bw.txt", bw);

    if(parser.check())
    {
        lablas::Matrix<T> check_matrix;
        check_matrix.set_preferred_matrix_format(CSR);
        check_matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);
        lablas::Vector<T> w_check(el.vertices_count);

        u.fill(1.0);
        w_check.fill(0.0);
        GrB_mxv(&w_check, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &check_matrix, &u, &desc);

        if(w == w_check)
        {
            cout << "Vectors are equal" << endl;
        }
        else
        {
            cout << "Vectors are NOT equal" << endl;
        }
    }

    #undef MASK_NULL
}

int main(int argc, char **argv) {
    try
    {
        test_spmv<float>(argc, argv);
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

