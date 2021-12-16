#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 11;
    avg_degree = 16;
    synthetic_graph_type = RMAT_GRAPH;
    storage_format = LAV;
    no_check = false;
    out_file_name = "kun_out.mtx";
    file_name = "lj.mtx";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::parse_args(int _argc, char **_argv)
{
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if ((option == "-scale") || (option == "-s"))
        {
            scale = atoi(_argv[++i]);
        }

        if ((option == "-out") || (option == "-outfile"))
        {
            out_file_name = string(_argv[++i]);
        }

        if((option == "-graph") || (option == "-type"))
        {
            option = _argv[++i];

            if ((option == "random_uniform") || (option == "ru") || (option == "RU"))
            {
                synthetic_graph_type = RANDOM_UNIFORM_GRAPH;
            }

            if ((option == "rmat") || (option == "RMAT"))
            {
                synthetic_graph_type = RMAT_GRAPH;
            }

            if ((option == "hpcg") || (option == "HPCG"))
            {
                synthetic_graph_type = HPCG_GRAPH;
            }

            if ((option == "real_world") || (option == "RW"))
            {
                synthetic_graph_type = REAL_WORLD_GRAPH;
                option = _argv[++i];
                file_name = string(option);
            }

            if ((option == "mtx") || (option == "MTX"))
            {
                synthetic_graph_type = MTX_GRAPH;
                option = _argv[++i];
                file_name = string(option);
            }
        }

        if ((option == "-edges") || (option == "-e"))
        {
            avg_degree = atoi(_argv[++i]);
        }
        
        if(option == "-format")
        {
            option = _argv[++i];
            
            if(option == "CSR")
                storage_format = CSR;
            else if(option == "COO")
                storage_format = COO;
            else if(option == "CSR_SEG")
                storage_format = CSR_SEG;
            else if(option == "LAV")
                storage_format = LAV;
            else if(option == "VG_CSR" || option == "vg_csr")
                storage_format = VECT_GROUP_CSR;
            else if(option == "SELL_C" || option == "SIGMA")
                storage_format = SELL_C;
        }

        if(option == "-no-check")
            no_check = true;
    }

    cout << "parser_stats: Format " << to_string(storage_format) << " GraphType " << to_string(synthetic_graph_type) << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
