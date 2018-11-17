#include "pch.h"

#pragma optimize("",off)

#include "tensorflow_lite_c_api.h"

#include <iostream>

#include <opencv2/opencv.hpp>

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

namespace opencv
{
    using mat = cv::Mat;

    mat resize(mat r, uint32_t w, uint32_t h)
    {
        mat m;

        cv::resize(r, m,cv::Size(w, h));
        return m;
    }

    auto width(mat r)
    {
        return r.cols;
    }

    auto height(mat r)
    {
        return r.rows;
    }

    auto byte_size(mat r)
    {
        return r.step[0] * r.rows;
    }

    auto normalize( mat r )
    {

        mat o0;
        r.convertTo(o0, CV_32FC3);

        //convert from 0-255 bytes to floats in the [-1;1]
        mat o1;
        o1 = (o0 - 127.5f) * (1.0f / 128.0f);
        return o1;
    }
}

namespace mtcnn
{
    const float threshold_pnet          = 0.8f;
    const float threshold_onet          = 0.8f;
    const float threshold_rnet          = 0.8f;

    const float minimum_face_size_px    = 20;
    const float initial_scale           = 0.7f;


    std::vector< float > make_scales(uint32_t width, uint32_t height, float min_face_size_px, float factor)
    {
        auto minl = std::min<float>(static_cast<float>(width), static_cast<float>(height));
        auto m = 12.0f / std::max<float>(min_face_size_px, 12.0f);

        //create scale pyramid
        std::vector<float> scales;

        auto factor_count = 0.0f;

        minl = minl * m;
        while (minl >= 12.0f)
        {
            auto scale = m * std::powf(factor, factor_count);
            scales.push_back(scale);

            minl = minl * factor;
            factor_count += 1.0f;
        }


        return scales;
    }

    struct model
    {
        tensorflow_lite_c_api::model                m_model;
        tensorflow_lite_c_api::interpreter_options  m_options;
        tensorflow_lite_c_api::interpreter          m_interpreter;
    };

    model make_model(const char* model_file)
    {
        tensorflow_lite_c_api::model                m(model_file);
        tensorflow_lite_c_api::interpreter_options  o;

        o.set_num_threads(8);
        tensorflow_lite_c_api::interpreter          i(m, o);
        return { std::move(m), std::move(o), std::move(i) };
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


