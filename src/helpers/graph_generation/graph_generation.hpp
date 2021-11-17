#pragma once

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

    double t1 = omp_get_wtime();
    RandomGenerator rng_api;
    int max_id_val = vertices_count;
    rng_api.generate_array_of_random_values<VNT>(src_ids, directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<VNT>(dst_ids, directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<T>(vals, directed_edges_count, 1.0);

    double t2 = omp_get_wtime();

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
void GraphGenerationAPI::R_MAT(EdgeListContainer<T> &_edges_container,
                               int _vertices_count,
                               long long _edges_count,
                               int _a_prob,
                               int _b_prob,
                               int _c_prob,
                               int _d_prob,
                               DirectionType _direction_type)
{
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    long long edges_count = _edges_count;

    int directed_edges_count = edges_count;
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
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge += step)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
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
            vals[cur_edge] = ((float) rand_r(&seed)) / (float) RAND_MAX;
            
            if(!_direction_type)
            {
                src_ids[cur_edge + 1] = to;
                dst_ids[cur_edge + 1] = from;
                vals[cur_edge + 1] = vals[cur_edge];
            }
        }
    }

    random_shuffle_edges(_edges_container);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

