#include "pch.h"

#include <memory>

#include "tensorflow_lite_c_api.h"

int32_t main(int32_t, char**)
{
    using namespace tensorflow_lite_c_api;

    model                m("data/hello_world.tflite");
    interpreter_options  o;
    
    o.set_num_threads(8);

    interpreter          i(m, o);

    auto in                                     = i.get_input_tensor_count();
    auto out                                    = i.get_output_tensor_count();

    auto t0                                     = input_tensor(i.get_input_tensor(0));
    auto t1                                     = output_tensor(i.get_output_tensor(0));

    i.allocate_tensors();

    float x = 2.0f;
    float y = 0.0f;

    t0.copy_from_buffer(&x, sizeof(x));

    i.invoke();
    
    t1.copy_to_buffer(&y, sizeof(y));
    
    return 0;
}
