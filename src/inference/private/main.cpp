#include "pch.h"

#include "tensorflow_c_api.h"


int32_t main(int32_t, char**)
{
    std::unique_ptr<tensorflow_c_api::session_options> m = std::make_unique<tensorflow_c_api::session_options>();
    return 0;
}
