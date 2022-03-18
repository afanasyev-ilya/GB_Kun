#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "parser_options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    VNT scale;
    VNT avg_degree;
    SyntheticGraphType synthetic_graph_type;
    MatrixStorageFormat storage_format;
    string file_name;
    string out_file_name;
    int iterations;

    bool no_check;

    string algo_name;
public:
    Parser();
    
    VNT get_scale() { return scale; };
    VNT get_avg_degree() { return avg_degree; };
    bool check()
    {
        return !no_check;
    };
    string get_file_name(){return file_name;};
    string get_out_file_name() {return out_file_name;};

    int get_iterations() {return iterations;};

    SyntheticGraphType get_synthetic_graph_type() {return synthetic_graph_type;};
    MatrixStorageFormat get_storage_format() {return storage_format;};

    void parse_args(int _argc, char **_argv);

    string get_algo_name() {return algo_name;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
