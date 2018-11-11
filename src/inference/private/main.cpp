#include "pch.h"

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

        auto placeholder = input_tensor(i.get_input_tensor(0));
        auto placeholder_1 = input_tensor(i.get_input_tensor(1));
        auto placeholder_2 = input_tensor(i.get_input_tensor(2));


        auto sz = placeholder.byte_size();
    }
}

namespace opencv
{
    using mat = cv::Mat;

    cv::Mat resize(mat r, uint32_t w, uint32_t h)
    {
        mat m;

        cv::resize(r, m,cv::Size(r.cols / 2, r.rows / 2));
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
}

//#pragma optimize("",off)
int32_t main(int32_t, char*[])
{ 
    using namespace cv;
    using namespace opencv;

    auto r  = imread("data/images/test1.jpg");

    auto w          = width(r);
    auto h          = height(r);

    auto scales     = mtcnn::make_scales(w, h, mtcnn::minimum_face_size_px, mtcnn::initial_scale);


    for (auto& v : scales)
    {
        std::cout << "Scale" << v << "\n";
    }


    return 0;
 }


