#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 10;
    avg_degree = 6;
    synthetic_graph_type = RMAT;
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

        if ((option == "-random_uniform") || (option == "-ru"))
        {
            synthetic_graph_type = RANDOM_UNIFORM;
        }

        if ((option == "-rmat") || (option == "-RMAT"))
        {
            synthetic_graph_type = RMAT;
        }

        if ((option == "-edges") || (option == "-e"))
        {
            avg_degree = atoi(_argv[++i]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
