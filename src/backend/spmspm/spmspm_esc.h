#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename SemiringT>
void SpMSpM_unmasked_esc(const Matrix<T> *_matrix1,
                         const Matrix<T> *_matrix2,
                         Matrix<T> *_matrix_result,
                         SemiringT _op)
{
    double t1 = omp_get_wtime();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    const auto n = _matrix1->get_csr()->get_num_rows();

    double t2 = omp_get_wtime();

    FILE *my_f;
    my_f = fopen("perf_stats.txt", "a");
    fprintf(my_f, "%s %lf (s) %lf (GFLOP/s) %lf (GB/s) %lld\n", "esc_mxm", (t2 - t1) * 1000, 0.0, 0.0, 0ll);
    fclose(my_f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
