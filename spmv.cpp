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

    lablas::Descriptor desc;

    lablas::Matrix<T> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix, parser);

    GrB_Index size;
    matrix.get_nrows(&size);
    lablas::Vector<T> w(size);
    lablas::Vector<T> u(size);

    #define MASK_NULL static_cast<const lablas::Vector<T>*>(NULL)

    vector<GrB_Index> nnz_subset;
    for(Index i = 0; i < ceil(size); i++)
        nnz_subset.push_back(rand() % size);

    u.fill(1.0);
    w.fill(1.0);
    GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, 10, &(nnz_subset[0]), nnz_subset.size(), NULL));
    GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);

    int num_runs = 10;
    double avg_time = 0;
    for(int run = 0; run < num_runs; run++)
    {
        w.fill(1.0);

        double t1 = omp_get_wtime();
        SAVE_STATS(GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);,
                   "SPMV", (sizeof(float)*2 + sizeof(size_t)), 1, &matrix);
        double t2 = omp_get_wtime();
        avg_time += (t2 - t1) / num_runs;
    }

    double perf = 2.0*matrix.get_nnz()/(avg_time*1e9);
    double bw = (2.0*sizeof(T)+sizeof(Index))*matrix.get_nnz()/(avg_time*1e9);
    cout << "SPMV avg time: " << avg_time*1000 << " ms" << endl;
    cout << "SPMV avg perf: " << perf << " GFlop/s" << endl;
    cout << "SPMV avg BW: " << bw << " GB/s" << endl;
    save_to_file("./output/perf.txt", perf);
    save_to_file("./output/bw.txt", bw);

    if(parser.check() && (parser.get_synthetic_graph_type() == MTX_GRAPH)) // can check only for external graph for now
    {
        lablas::Matrix<T> check_matrix;
        check_matrix.set_preferred_matrix_format(CSR);
        init_matrix(check_matrix, parser);

        lablas::Vector<T> w_check(size);

        u.fill(1.0);
        w_check.fill(1.0);
        GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, 10, &(nnz_subset[0]), nnz_subset.size(), NULL));

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
        print_omp_stats();
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

