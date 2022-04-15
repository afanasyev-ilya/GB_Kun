#include "src/gb_kun.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        lablas::Descriptor desc;

        if(argc != 2)
        {
            std::cout << "file name must be provided" << std::endl;
            return 0;
        }
        std::string file_name = argv[1];
        lablas::Matrix<float> txt_matrix;
        txt_matrix.init_from_mtx(file_name + ".mtx");

        lablas::Matrix<float> bin_matrix;
        bin_matrix.init_from_mtx(file_name + ".mtxbin");

        if(bin_matrix == txt_matrix)
            std::cout << "txt and binary graphs ARE equal!" << std::endl;
        else
            std::cout << "txt and binary graphs ARE NOT equal!" << std::endl;
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

