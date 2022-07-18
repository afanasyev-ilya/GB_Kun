#ifdef __USE_GTEST__
#include <gtest/gtest.h>
#endif

#include "src/gb_kun.h"


int main(int argc, char **argv) {
    try
    {
        // run unit tests here
    }
    catch (string& error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}
