#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 10;
    avg_degree = 16;
    synthetic_graph_type = RMAT;
    storage_format = CSR;
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
                synthetic_graph_type = RANDOM_UNIFORM;
            }

            if ((option == "rmat") || (option == "RMAT"))
            {
                synthetic_graph_type = RMAT;
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
        }

        if(option == "-no-check")
            no_check = true;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
