#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define EDGE_VAL ((float) rand_r(&seed)) / (float) RAND_MAX
#define EDGE_VAL 1

template <typename T>
void GraphGenerationAPI::generate_synthetic_graph(EdgeListContainer<T> &_edges_container, Parser &_parser)
{
    VNT scale = _parser.get_scale();
    VNT avg_deg = _parser.get_avg_degree();

    if(_parser.get_synthetic_graph_type() == RANDOM_UNIFORM_GRAPH)
    {
        GraphGenerationAPI::random_uniform(_edges_container,
                                           pow(2.0, scale),
                                           avg_deg * pow(2.0, scale));
    }
    else if(_parser.get_synthetic_graph_type() == RMAT_GRAPH)
    {
        GraphGenerationAPI::RMAT(_edges_container, pow(2.0, scale), avg_deg * pow(2.0, scale), 57, 19, 19, 5);
    }
    else if(_parser.get_synthetic_graph_type() == HPCG_GRAPH)
    {
        GraphGenerationAPI::HPCG(_edges_container, scale, scale, scale, avg_deg);
    }
    else if(_parser.get_synthetic_graph_type() == MTX_GRAPH)
    {
        GraphGenerationAPI::init_from_mtx_file(_edges_container, _parser.get_file_name());
    }
    else if(_parser.get_synthetic_graph_type() == REAL_WORLD_GRAPH)
    {
        GraphGenerationAPI::init_from_txt_file(_edges_container, _parser.get_file_name(), DIRECTED_GRAPH);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void random_shuffle_edges(EdgeListContainer<T> &_edges_container)
{
    srand ( unsigned ( time(0) ) );

    VNT vertices_count = _edges_container.vertices_count;
    ENT edges_count = _edges_container.edges_count;
    VNT *src_ids = _edges_container.src_ids.data();
    VNT *dst_ids = _edges_container.dst_ids.data();
    T *vals = _edges_container.edge_vals.data();

    vector<VNT> reorder_ids(vertices_count);
    #pragma omp parallel for
    for (VNT i = 0; i < vertices_count; i++)
        reorder_ids[i] = i;

    random_shuffle(reorder_ids.begin(), reorder_ids.end() );

    for(ENT i = 0; i < edges_count; i++)
    {
        src_ids[i] = reorder_ids[src_ids[i]];
        dst_ids[i] = reorder_ids[dst_ids[i]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GraphGenerationAPI::random_uniform(EdgeListContainer<T> &_edges_container,
                                        VNT _vertices_count,
                                        ENT _edges_count,
                                        DirectionType _direction_type)
{
    cout << "Creating Random Uniform matrix" << endl;
    VNT vertices_count = _vertices_count;
    ENT edges_count = _edges_count;
    
    ENT directed_edges_count = edges_count;
    if(!_direction_type)
        edges_count *= 2;

    _edges_container.vertices_count = vertices_count;
    _edges_container.edges_count = edges_count;
    _edges_container.src_ids.resize(edges_count);
    _edges_container.dst_ids.resize(edges_count);
    _edges_container.edge_vals.resize(edges_count);

    // get pointers
    VNT *src_ids = _edges_container.src_ids.data();
    VNT *dst_ids = _edges_container.dst_ids.data();
    T *vals = _edges_container.edge_vals.data();

    #pragma omp parallel
    {};

    RandomGenerator rng_api;
    int max_id_val = vertices_count;
    rng_api.generate_array_of_random_values<VNT>(src_ids, directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<VNT>(dst_ids, directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<T>(vals, directed_edges_count, 1.0);

    if(!_direction_type)
    {
        #pragma omp parallel for
        for(ENT i = 0; i < directed_edges_count; i++)
        {
            src_ids[i + directed_edges_count] = dst_ids[i];
            dst_ids[i + directed_edges_count] = src_ids[i];
            vals[i + directed_edges_count] = vals[i];
        }
    }

    random_shuffle_edges(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GraphGenerationAPI::RMAT(EdgeListContainer<T> &_edges_container,
                              VNT _vertices_count,
                              ENT _edges_count,
                              int _a_prob,
                              int _b_prob,
                              int _c_prob,
                              int _d_prob,
                              DirectionType _direction_type)
{
    cout << "Creating RMAT matrix" << endl;
    VNT n = (VNT)log2(_vertices_count);
    VNT vertices_count = _vertices_count;
    ENT edges_count = _edges_count;

    ENT directed_edges_count = edges_count;
    if(!_direction_type)
        edges_count *= 2;
    
    int step = 1;
    if(!_direction_type)
        step = 2;

    _edges_container.vertices_count = vertices_count;
    _edges_container.edges_count = edges_count;
    _edges_container.src_ids.resize(edges_count);
    _edges_container.dst_ids.resize(edges_count);
    _edges_container.edge_vals.resize(edges_count);

    // get pointers
    VNT *src_ids = _edges_container.src_ids.data();
    VNT *dst_ids = _edges_container.dst_ids.data();
    T *vals = _edges_container.edge_vals.data();

    int threads_count = omp_get_max_threads();
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel private(seed) num_threads(threads_count)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(guided, 1024)
        for (ENT cur_edge = 0; cur_edge < edges_count; cur_edge += step)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (ENT i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
                
                int step = (int)pow(2, n - (i + 1));
                
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
            
            int from = x_middle;
            int to = y_middle;

            src_ids[cur_edge] = from;
            dst_ids[cur_edge] = to;
            vals[cur_edge] = EDGE_VAL;
            
            if(!_direction_type)
            {
                src_ids[cur_edge + 1] = to;
                dst_ids[cur_edge + 1] = from;
                vals[cur_edge + 1] = vals[cur_edge];
            }
        }
    }

    random_shuffle_edges(_edges_container);
    //save_to_file(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GraphGenerationAPI::HPCG(EdgeListContainer<T> &_edges_container,
                               VNT _nx, VNT _ny, VNT _nnz, ENT _edge_factor)
{
    cout << "Creating HPCG matrix" << endl;
    int numberOfNonnzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil
    int numberOfRows = _nx*_ny*_nnz;

    int size = numberOfRows;
    int nnz = numberOfRows * numberOfNonnzerosPerRow;

    // Allocate arrays that are of length localNumberOfRows
    int * nonnzerosInRow = new int[numberOfRows];
    int ** mtxInd = new int*[numberOfRows];
    T ** matrixValues = new T*[numberOfRows];

    int *col_ = new int[numberOfNonnzerosPerRow*numberOfRows];
    T *val_ = new T[sizeof(T)*numberOfNonnzerosPerRow*numberOfRows];

    int* boundaryRows = new int[_nx*_ny*_nnz - (_nx-2)*(_ny-2)*(_nnz-2)];

    if ( col_ == NULL || val_ == NULL || nonnzerosInRow == NULL || boundaryRows == NULL
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
        for (int z = 1; z < _nnz - 1; z++) {
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
                boundaryRows[_ny*_nx + 2*(_nnz-2)*(_nx+_ny-2) + y*_nx + x] = ((_nnz - 1)*_ny + y)*_nx + x;
                numOfBoundaryRows++;
            }
        }
    }

    int numberOfNonnzeros = 0;

    #pragma omp parallel reduction(+:numberOfNonnzeros)
    {
        int ithr = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        int works = (_nnz - 2)*(_ny - 2);
        int begin = ((ithr  )*works)/nthr;
        int end   = ((ithr+1)*works)/nthr;
        for (int i = begin; i < end; i++)
        {
            int iz = i/(_ny - 2) + 1;
            int iy = i%(_ny - 2) + 1;

            for (int ix=1; ix<_nx-1; ix++)
            {
                int currentLocalRow = iz*_nx*_ny+iy*_nx+ix;
                mtxInd[currentLocalRow]      = col_ + currentLocalRow*numberOfNonnzerosPerRow;
                matrixValues[currentLocalRow] = val_ + currentLocalRow*numberOfNonnzerosPerRow;
                char numberOfNonnzerosInRow = 0;
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
                numberOfNonnzerosInRow += 27;
                nonnzerosInRow[currentLocalRow] = numberOfNonnzerosInRow;
                numberOfNonnzeros += numberOfNonnzerosInRow; // Protect this with an atomic
            } // end ix loop
        }

        #pragma omp for
        for (int i = 0; i < numOfBoundaryRows; i++) {
            int currentLocalRow = boundaryRows[i];

            int iz = currentLocalRow/(_ny*_nx);
            int iy = currentLocalRow/_nx%_ny;
            int ix = currentLocalRow%_nx;

            int sz_begin = std::max<int>(-1, -iz);
            int sz_end = std::min<int>(1, _nnz - iz - 1);

            int sy_begin = std::max<int>(-1, -iy);
            int sy_end = std::min<int>(1, _ny - iy - 1);

            int sx_begin = std::max<int>(-1, -ix);
            int sx_end = std::min<int>(1, _nx - ix - 1);


            mtxInd[currentLocalRow]      = col_ + currentLocalRow*numberOfNonnzerosPerRow;
            matrixValues[currentLocalRow] = val_ + currentLocalRow*numberOfNonnzerosPerRow;
            char numberOfNonnzerosInRow = 0;
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
                        numberOfNonnzerosInRow++;
                    } // end sx loop
                } // end sy loop
            } // end sz loop
            nonnzerosInRow[currentLocalRow] = numberOfNonnzerosInRow;
            numberOfNonnzeros += numberOfNonnzerosInRow; // Protect this with an atomic
        }
    }

    _edges_container.vertices_count = numberOfRows;
    _edges_container.edges_count = numberOfNonnzeros;
    _edges_container.src_ids.resize(numberOfNonnzeros);
    _edges_container.dst_ids.resize(numberOfNonnzeros);
    _edges_container.edge_vals.resize(numberOfNonnzeros);

    // get pointers
    VNT *src_ids = _edges_container.src_ids.data();
    VNT *dst_ids = _edges_container.dst_ids.data();
    T *vals = _edges_container.edge_vals.data();

    long long k = 0;
    for(int i=0; i<numberOfRows; ++i)
    {
        for(int j=0; j<nonnzerosInRow[i];++j)
        {
            src_ids[k] = i;
            dst_ids[k] = (mtxInd[i])[j];
            vals[k] = (matrixValues[i])[j];
            ++k;
        }
    }


    delete [] col_;
    delete [] val_;
    delete [] mtxInd;
    delete [] matrixValues;
    delete [] nonnzerosInRow;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GraphGenerationAPI::init_from_txt_file(EdgeListContainer<T> &_edges_container, string _txt_file_name,
                                            DirectionType _direction_type)
{
    ifstream infile(_txt_file_name.c_str());
    if (!infile.is_open())
        throw "can't open file during convert";

    int vertices_count = 0;
    long long edges_count = 0;
    string line;
    getline(infile, line); // read first line

    for(int i = 0; i < 5;i++)
        getline(infile, line);

    vector<int>tmp_src_ids;
    vector<int>tmp_dst_ids;

    long long i = 0;
    while (getline(infile, line))
    {
        istringstream iss(line);
        int src_id = 0, dst_id = 0;
        if (!(iss >> src_id >> dst_id))
        {
            continue;
        }

        if(src_id >= vertices_count)
            vertices_count = src_id + 1;

        if(dst_id >= vertices_count)
            vertices_count = dst_id + 1;

        tmp_src_ids.push_back(src_id);
        tmp_dst_ids.push_back(dst_id);
        i++;

        if(_direction_type == UNDIRECTED_GRAPH)
        {
            tmp_src_ids.push_back(dst_id);
            tmp_dst_ids.push_back(src_id);
            i++;
        }
    }

    cout << "direction type: " << _direction_type << endl;
    cout << "loaded " << vertices_count << " vertices_count" << endl;
    if(_direction_type == DIRECTED_GRAPH)
        cout << "loaded " << i << " edges" << endl;
    else
        cout << "loaded " << i << " directed edges, " << i/2 << " undirected" << endl;

    edges_count = i;

    _edges_container.vertices_count = vertices_count;
    _edges_container.edges_count = edges_count;
    _edges_container.src_ids.resize(edges_count);
    _edges_container.dst_ids.resize(edges_count);
    _edges_container.edge_vals.resize(edges_count);

    unsigned int seed = int(time(NULL));
    for(i = 0; i < edges_count; i++)
    {
        _edges_container.src_ids[i] = tmp_src_ids[i];
        _edges_container.dst_ids[i] = tmp_dst_ids[i];
        _edges_container.edge_vals[i] = ((float) rand_r(&seed)) / (float) RAND_MAX;
    }

    // validate
    for(i = 0; i < edges_count; i++)
    {
        int src_id = _edges_container.src_ids[i];
        int dst_id = _edges_container.dst_ids[i];
        if((src_id >= vertices_count) || (src_id < 0))
        {
            cout << "error src: " << src_id << endl;
            throw "Error: incorrect src id on conversion";
        }
        if((dst_id >= vertices_count) || (dst_id < 0))
        {
            cout << "error dst: " << dst_id << endl;
            throw "Error: incorrect dst id on conversion";
        }
    }

    infile.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GraphGenerationAPI::init_from_mtx_file(EdgeListContainer<T> &_edges_container, string _mtx_file_name)
{
    cout << "reading from mtx file" << endl;
    ifstream infile(_mtx_file_name.c_str());
    if (!infile.is_open())
        throw "can't open file during convert";

    VNT vertices_count = 0;
    ENT edges_count = 0;
    string line;
    getline(infile, line); // read first line

    if (line.find("%%MatrixMarket matrix coordinate pattern general") == std::string::npos)
    {
        throw "Error: is not a mtx file";
    }

    getline(infile, line);

    istringstream iss(line);
    VNT rows = 0; VNT cols = 0; ENT nnz = 0;
    iss >> rows >> cols >> nnz;

    if(rows != cols)
    {
        throw "Error: is not a square matrix";
    }

    _edges_container.vertices_count = rows;
    _edges_container.edges_count = nnz;
    _edges_container.src_ids.resize(nnz);
    _edges_container.dst_ids.resize(nnz);
    _edges_container.edge_vals.resize(nnz);

    ENT i = 0;
    unsigned int seed = int(time(NULL));
    while (getline(infile, line))
    {
        istringstream iss(line);
        VNT src_id = 0, dst_id = 0;
        iss >> src_id >> dst_id;

        _edges_container.src_ids[i] = src_id - 1;
        _edges_container.dst_ids[i] = dst_id - 1;
        _edges_container.edge_vals[i] = EDGE_VAL;
        i++;
    }

    infile.close();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
