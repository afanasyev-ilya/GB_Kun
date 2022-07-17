#include "src/gb_kun.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void save_to_file(const string &_file_name, double _stat)
{
    ofstream stat_file;
    stat_file.open(_file_name, std::ios_base::app);
    stat_file << _stat << endl;
    stat_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void report_num_threads(int level)
{
    #pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
               level, omp_get_num_threads());
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void check_mxv(lablas::Vector<T> &_out, lablas::Matrix<T> &_matrix, lablas::Vector<T> &_in)
{
    T* out_vals = _out.get_vector()->getDense()->get_vals();
    T* in_vals = _in.get_vector()->getDense()->get_vals();
    lablas::backend::MatrixCSR<T> *csr_matrix = ((lablas::backend::MatrixCSR<T> *) _matrix.get_matrix()->get_csr());

    Index num_rows;
    _matrix.get_nrows(&num_rows);
    for(VNT row = 0; row < num_rows; row++)
    {
        T res = 0;
        for(ENT j = csr_matrix->get_row_ptr()[row]; j < csr_matrix->get_row_ptr()[row + 1]; j++)
        {
            VNT col = csr_matrix->get_col_ids()[j];
            T val = csr_matrix->get_vals()[j];
            res = res + val * in_vals[col];
        }
        out_vals[row] = res;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    Index sparsity_k = 1000.0;
    vector<GrB_Index> nnz_subset;
    for(Index i = 0; i < size/sparsity_k + 1; i++)
        nnz_subset.push_back(rand() % size);

    desc.set(GrB_NEON, GrB_NEON_ON);

    Index u_const = 1;
    Index u_diff_const = 5;
    u.fill(u_const);
    w.fill(1.0);
    GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, u_diff_const, &(nnz_subset[0]), nnz_subset.size(), NULL));
    GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);

    if (!parser.check()) {
         int num_runs = parser.get_iterations();
         double avg_time = 0;
         for (int run = 0; run < num_runs; run++) {
             w.fill(1.0);

             double t1 = omp_get_wtime();
             SAVE_STATS(GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);,
                        "SPMV", (sizeof(float) * 2 + sizeof(size_t)), 1, &matrix);
             double t2 = omp_get_wtime();
             avg_time += (t2 - t1) / num_runs;
         }

         double perf = 2.0 * matrix.get_nnz() / (avg_time * 1e9);
         double bw = (2.0 * sizeof(T) + sizeof(Index)) * matrix.get_nnz() / (avg_time * 1e9);
         cout << "SPMV avg time: " << avg_time * 1000 << " ms" << endl;
         cout << "SPMV avg perf: " << perf << " GFlop/s" << endl;
         cout << "SPMV avg BW: " << bw << " GB/s" << endl;
         save_to_file("./output/perf.txt", perf);
         save_to_file("./output/bw.txt", bw);

    }

    if (parser.check()) // can check only for external graph for now
    {
        lablas::Vector<T> w_check(size);

        u.fill(u_const);
        w_check.fill(1.0);
        GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, u_diff_const, &(nnz_subset[0]), nnz_subset.size(), NULL));

        check_mxv(w_check, matrix, u);

        if(w == w_check)
        {
            print_diff(w, w_check);
            cout << "Vectors are equal" << endl;
        }
        else
        {
            print_diff(w, w_check);
            cout << "Vectors are NOT equal" << endl;
        }
    }

    #undef MASK_NULL
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        test_spmv<long int>(argc, argv);
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

