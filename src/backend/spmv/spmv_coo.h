#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
double SpMV(MatrixCOO<T> &_A, DenseVector<T> &_x, DenseVector<T> &_y)
{
    ENT nz = _A.get_nz();
    VNT *row_ids = _A.get_row_ids();
    VNT *col_ids = _A.get_col_ids();
    T *vals = _A.get_vals();

    T *x_vals = _x.get_vals();
    T *y_vals = _y.get_vals();

    double t1 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for(ENT i = 0; i < nz; i++)
    {
        VNT row = row_ids[i];
        VNT col = col_ids[i];
        T val = vals[i];
        #pragma omp atomic
        y_vals[row] += val * x_vals[col];
    }

    double t2 = omp_get_wtime();

    cout << "SPMV(COO) perf: " << 2.0*_A.get_nz()/((t2-t1)*1e9) << " GFlop/s" << endl;
    double bw = (3.0*sizeof(T)+2*sizeof(VNT))*_A.get_nz()/((t2-t1)*1e9);
    cout << "SPMV(COO) bw: " << bw << " GB/s" << endl;
    return bw;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
