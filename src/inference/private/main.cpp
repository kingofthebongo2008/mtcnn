#include "pch.h"

#include "tensorflow_c_api.h"


int32_t main(int32_t, char**)
{
    using namespace tensorflow_c_api;
    using namespace std;

    unique_ptr<session_options> m = make_unique<session_options>();
    unique_ptr<graph>           g = make_unique<graph>();
    return 0;
}
