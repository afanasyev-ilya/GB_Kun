#include "src/gb_kun.h"

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        EdgeListContainer<float> el;
        GraphGenerationAPI::generate_synthetic_graph(el, parser);

        el.save_as_mtx(parser.get_out_file_name());
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

