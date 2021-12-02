#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 16;
    avg_degree = 27;
    synthetic_graph_type = RANDOM_UNIFORM_GRAPH;
    storage_format = SELL_C;
    no_check = false;
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
            else if(option == "COO_OPT")
                storage_format = COO_OPT;
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
