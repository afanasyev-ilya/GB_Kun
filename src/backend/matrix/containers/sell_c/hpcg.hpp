namespace lablas {
    namespace backend {

        template<typename T>
        void MatrixSellC<T>::generateHPCG(int _nx, int _ny, int _nz)
        {
            cout << "doing hpcg matrix" << endl;
            int numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil
            int numberOfRows = _nx*_ny*_nz;

            size = numberOfRows;
            nz = numberOfRows * numberOfNonzerosPerRow;

            // Allocate arrays that are of length localNumberOfRows
            int * nonzerosInRow = new int[numberOfRows];
            int ** mtxInd = new int*[numberOfRows];
            T ** matrixValues = new T*[numberOfRows];

            int *col_ = new int[numberOfNonzerosPerRow*numberOfRows];
            T *val_ = new T[sizeof(T)*numberOfNonzerosPerRow*numberOfRows];

            int* boundaryRows = new int[_nx*_ny*_nz - (_nx-2)*(_ny-2)*(_nz-2)];

            if ( col_ == NULL || val_ == NULL || nonzerosInRow == NULL || boundaryRows == NULL
                 || mtxInd == NULL || matrixValues == NULL)
            {
                return;
            }

            int numOfBoundaryRows = 0;
#pragma omp parallel reduction(+:numOfBoundaryRows)
            {
#pragma omp for nowait
                for (int y = 0; y < _ny; y++) {
                    for (int x = 0; x < _nx; x++) {
                        boundaryRows[y*_nx + x] = y*_nx + x;
                        numOfBoundaryRows++;
                    }
                }

#pragma omp for nowait
                for (int z = 1; z < _nz - 1; z++) {
                    for (int x = 0; x < _nx; x++) {
                        boundaryRows[_ny*_nx + 2*(z-1)*(_nx+_ny-2) + x ] = z*_ny*_nx + x;
                        numOfBoundaryRows++;
                    }
                    for (int y = 1; y < _ny - 1; y++) {
                        boundaryRows[_ny*_nx + 2*(z-1)*(_nx+_ny-2) + _nx + 2*(y-1)] = (z*_ny + y)*_nx;
                        numOfBoundaryRows++;
                        boundaryRows[_ny*_nx + 2*(z-1)*(_nx+_ny-2) + _nx + 2*(y-1)+1] = (z*_ny + y)*_nx + _nx - 1;
                        numOfBoundaryRows++;
                    }
                    for (int x = 0; x < _nx; x++) {
                        boundaryRows[_ny*_nx + 2*(z-1)*(_nx+_ny-2) + _nx + 2*(_ny-2) + x] = (z*_ny + (_ny - 1))*_nx + x;
                        numOfBoundaryRows++;
                    }
                }

#pragma omp for nowait
                for (int y = 0; y < _ny; y++) {
                    for (int x = 0; x < _nx; x++) {
                        boundaryRows[_ny*_nx + 2*(_nz-2)*(_nx+_ny-2) + y*_nx + x] = ((_nz - 1)*_ny + y)*_nx + x;
                        numOfBoundaryRows++;
                    }
                }
            }

            int numberOfNonzeros = 0;

#pragma omp parallel reduction(+:numberOfNonzeros)
            {
                int ithr = omp_get_thread_num();
                int nthr = omp_get_num_threads();

                int works = (_nz - 2)*(_ny - 2);
                int begin = ((ithr  )*works)/nthr;
                int end   = ((ithr+1)*works)/nthr;
                for (int i = begin; i < end; i++)
                {
                    int iz = i/(_ny - 2) + 1;
                    int iy = i%(_ny - 2) + 1;

                    for (int ix=1; ix<_nx-1; ix++)
                    {
                        int currentLocalRow = iz*_nx*_ny+iy*_nx+ix;
                        mtxInd[currentLocalRow]      = col_ + currentLocalRow*numberOfNonzerosPerRow;
                        matrixValues[currentLocalRow] = val_ + currentLocalRow*numberOfNonzerosPerRow;
                        char numberOfNonzerosInRow = 0;
                        T * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
                        int  * currentIndexPointerL = mtxInd[currentLocalRow]; // Pointer to current index in current row
                        for (int sz=-1; sz<=1; sz++) {

                            *(currentValuePointer + 0) = -1.0;
                            *(currentValuePointer + 1) = -1.0;
                            *(currentValuePointer + 2) = -1.0;
                            *(currentValuePointer + 3) = -1.0;
                            *(currentValuePointer + 4) = -1.0;
                            *(currentValuePointer + 5) = -1.0;
                            *(currentValuePointer + 6) = -1.0;
                            *(currentValuePointer + 7) = -1.0;
                            *(currentValuePointer + 8) = -1.0;

                            int offset = currentLocalRow + sz*_ny*_nx;
                            *(currentIndexPointerL + 0) = offset - _nx - 1;
                            *(currentIndexPointerL + 1) = offset - _nx;
                            *(currentIndexPointerL + 2) = offset - _nx + 1;
                            *(currentIndexPointerL + 3) = offset - 1;
                            *(currentIndexPointerL + 4) = offset;
                            *(currentIndexPointerL + 5) = offset + 1;
                            *(currentIndexPointerL + 6) = offset + _nx - 1;
                            *(currentIndexPointerL + 7) = offset + _nx;
                            *(currentIndexPointerL + 8) = offset + _nx + 1;

                            currentValuePointer  += 9;
                            currentIndexPointerL += 9;
                        } // end sz loop
                        *(currentValuePointer - 14) = 26.0;
                        numberOfNonzerosInRow += 27;
                        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
                        numberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
                    } // end ix loop
                }

#pragma omp for
                for (int i = 0; i < numOfBoundaryRows; i++) {
                    int currentLocalRow = boundaryRows[i];

                    int iz = currentLocalRow/(_ny*_nx);
                    int iy = currentLocalRow/_nx%_ny;
                    int ix = currentLocalRow%_nx;

                    int sz_begin = std::max<int>(-1, -iz);
                    int sz_end = std::min<int>(1, _nz - iz - 1);

                    int sy_begin = std::max<int>(-1, -iy);
                    int sy_end = std::min<int>(1, _ny - iy - 1);

                    int sx_begin = std::max<int>(-1, -ix);
                    int sx_end = std::min<int>(1, _nx - ix - 1);


                    mtxInd[currentLocalRow]      = col_ + currentLocalRow*numberOfNonzerosPerRow;
                    matrixValues[currentLocalRow] = val_ + currentLocalRow*numberOfNonzerosPerRow;
                    char numberOfNonzerosInRow = 0;
                    T * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
                    int  * currentIndexPointerL = mtxInd[currentLocalRow];
                    for (int sz=sz_begin; sz<=sz_end; sz++) {
                        for (int sy=sy_begin; sy<=sy_end; sy++) {
                            for (int sx=sx_begin; sx<=sx_end; sx++) {
                                int    col = currentLocalRow + sz*_nx*_ny+sy*_nx+sx;
                                if (col==currentLocalRow) {
                                    *currentValuePointer++ = 26.0;
                                } else {
                                    *currentValuePointer++ = -1.0;
                                }
                                *currentIndexPointerL++ = col;
                                numberOfNonzerosInRow++;
                            } // end sx loop
                        } // end sy loop
                    } // end sz loop
                    nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
                    numberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
                }
            }

            row_ptr = new int[numberOfRows+1];
            col_ids = new int[numberOfNonzeros];
            vals = new T[numberOfNonzeros];

            row_ptr[0] = 0;
#pragma omp parallel for
            for(int i=0; i<numberOfRows; ++i)
            {
                row_ptr[i+1] = nonzerosInRow[i];
            }

            for(int i=0; i<numberOfRows; ++i)
            {
                row_ptr[i+1] += row_ptr[i];
            }

#pragma omp parallel for
            for(int i=0; i<numberOfRows; ++i)
            {
                int k = row_ptr[i];
                for(int j=0; j<nonzerosInRow[i];++j)
                {
                    col_ids[k] = (mtxInd[i])[j];
                    vals[k] = (matrixValues[i])[j];
                    ++k;
                }
            }

            size = numberOfRows;
            nz = numberOfNonzeros;

            delete [] col_;
            delete [] val_;
            delete [] mtxInd;
            delete [] matrixValues;
            nz_per_row = nonzerosInRow;
//    free(nonzerosInRow);

            return;
        }

    }
}