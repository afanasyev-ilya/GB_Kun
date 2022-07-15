#include "src/gb_kun.h"
#include <cstdlib>
#define nested_param 10

template<typename T>
void test_nested_spmv(int argc, char **argv)
{
    setenv("OMP_NESTED", "FALSE", 1);
    Parser parser;
    parser.parse_args(argc, argv);

    double sequential_runs[nested_param];

    lablas::Descriptor desc;

#define MASK_NULL static_cast<const lablas::Vector<T>*>(NULL)


    int num_runs = parser.get_iterations();

    double repeating_start = omp_get_wtime();

    for (int iteration = 0; iteration < nested_param; iteration++) {

        lablas::Matrix<T> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        GrB_Index size;
        matrix.get_nrows(&size);
        lablas::Vector<T> w(size);
        lablas::Vector<T> u(size);


        Index sparsity_k = 1000.0;
        vector <GrB_Index> nnz_subset;
        for (Index i = 0; i < size / sparsity_k + 1; i++)
            nnz_subset.push_back(rand() % size);

        Index u_const = 1;
        Index u_diff_const = 5;
        u.fill(u_const);
        GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, u_diff_const, &(nnz_subset[0]), nnz_subset.size(), NULL));
        /* We don't need to do a heating run */

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
    }

    double repeating_end = omp_get_wtime();

    std::cout << "Overall sequential run time is " << repeating_end - repeating_start << std::endl;
    std::cout << "Average sequential run time is " << (repeating_end - repeating_start) / nested_param << std::endl;

    setenv("OMP_NESTED", "TRUE", 1);
    printf("OMP_NESTED = %s\n", getenv("OMP_NESTED"));
    double nested_start = omp_get_wtime();

/* Following region is being run by N threads, each of the threads is having own matrix and vector "copy" */
#pragma omp parallel num_threads(nested_param)
{

    lablas::Matrix<T> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix, parser);

    GrB_Index size;
    matrix.get_nrows(&size);
    lablas::Vector<T> w(size);
    lablas::Vector<T> u(size);


    Index sparsity_k = 1000.0;
    vector <GrB_Index> nnz_subset;
    for (Index i = 0; i < size / sparsity_k + 1; i++)
        nnz_subset.push_back(rand() % size);

    Index u_const = 1;
    Index u_diff_const = 5;
    u.fill(u_const);
    GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, u_diff_const, &(nnz_subset[0]), nnz_subset.size(), NULL));
    /* We don't need to do a heating run */

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

#pragma omp critical
    {
        sequential_runs[omp_get_thread_num()] = avg_time;
    }
}
    double nested_end = omp_get_wtime();

    double max_time = 0.0, min_time = INT32_MAX, avg_time = 0.0;
    for (int iter = 0; iter < nested_param; iter++) {
        avg_time += sequential_runs[iter];
        if (sequential_runs[iter] > max_time) {
            max_time = sequential_runs[iter];
        }
        if (sequential_runs[iter] < min_time) {
            min_time = sequential_runs[iter];
        }
    }
    avg_time /= nested_param;

    std::cout << "Overall nested time is " << nested_end - nested_start << std::endl;
    std::cout << "Deviation+ = " << max_time-avg_time <<", Deviation- = " << avg_time-min_time << std::endl;
    std::cout << "Average thread running time = " << avg_time << std::endl;

#undef MASK_NULL
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        test_nested_spmv<float>(argc, argv);
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


