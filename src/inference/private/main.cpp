#include "pch.h"

#pragma optimize("",off)

#include <iostream>

#include "opencv_bridge.h"
#include "mtcnn_bridge.h"

namespace
{
    void test_hello_world()
    {
        using namespace tensorflow_lite_c_api;

        model                m("data/hello_world.tflite");
        interpreter_options  o;

        o.set_num_threads(8);

        interpreter          i(m, o);

        auto in = i.get_input_tensor_count();
        auto out = i.get_output_tensor_count();

        auto t0 = input_tensor(i.get_input_tensor(0));
        auto t1 = output_tensor(i.get_output_tensor(0));

        i.allocate_tensors();

        float x = 2.0f;
        float y = 0.0f;

        t0.copy_from_buffer(&x, sizeof(x));

        i.invoke();

        t1.copy_to_buffer(&y, sizeof(y));

        std::cout << "Result is " << y << std::endl;
    }

    void test_load_mtcnn_model()
    {
        using namespace tensorflow_lite_c_api;

        model                m("data/mtcnn.tflite");
        interpreter_options  o;

        o.set_num_threads(8);

        interpreter          i(m, o);

        auto in = i.get_input_tensor_count();
        auto out = i.get_output_tensor_count();

        auto placeholder    = input_tensor(i.get_input_tensor(0));
        auto placeholder_1  = input_tensor(i.get_input_tensor(1));
        auto placeholder_2  = input_tensor(i.get_input_tensor(2));


        auto sz = placeholder.byte_size();
    }
}


int32_t main(int32_t, char*[])
{ 
    auto r          = cv::imread("data/images/test1.jpg");

    auto w          = opencv::width(r);
    auto h          = opencv::height(r);
    auto scales     = mtcnn::make_scales(w, h, mtcnn::minimum_face_size_px, mtcnn::initial_scale);

    auto m          = mtcnn::make_model("data/mtcnn.tflite");
    

    m.m_interpreter.allocate_tensors();

    for (auto& v : scales)
    {
        auto ws                     = std::ceilf(w * v);
        auto hs                     = std::ceilf(h * v);
        auto img0                   = opencv::normalize(r);

        auto pnet                   = tensorflow_lite_c_api::make_input_tensor(&m.m_interpreter, 0);

        auto d                      = pnet.num_dims();
        int32_t size[4]             = { 1, ws, hs, 3 };
        pnet.copy_from_buffer(img0.data, opencv::byte_size(img0));

        m.m_interpreter.invoke();
    }


    return 0;
 }


