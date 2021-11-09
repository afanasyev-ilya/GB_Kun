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
        
        if ((option.compare("-scale") == 0) || (option.compare("-s") == 0))
        {
            scale = atoi(_argv[++i]);
        }

        if ((option.compare("-random_uniform") == 0) || (option.compare("-ru") == 0))
        {
            synthetic_graph_type = RANDOM_UNIFORM;
        }

        if ((option.compare("-rmat") == 0) || (option.compare("-RMAT") == 0))
        {
            synthetic_graph_type = RMAT;
        }

        if ((option.compare("-edges") == 0) || (option.compare("-e") == 0))
        {
            avg_degree = atoi(_argv[++i]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
