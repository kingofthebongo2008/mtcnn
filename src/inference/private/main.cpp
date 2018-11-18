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
}

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace mtcnn
{
    struct numpy_wrapper
    {
        std::vector<float>  m_data;
        xt::xarray<float>   m_numpy;
    };

    auto make_xtensor_4(const tensorflow_lite_c_api::output_tensor& tensor)
    {
        std::array<size_t, 4> shape = { tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3) };
        auto size_in_floats         = tensor.byte_size() / sizeof(float);

        numpy_wrapper w;
        w.m_data.resize(size_in_floats);

        tensor.copy_to_buffer(&w.m_data[0], tensor.byte_size() );
        w.m_numpy = xt::adapt(w.m_data, shape);

        return w;
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
    auto ot = m.m_interpreter.get_output_tensor_count();
    if (true)
    {
        for (auto& v : scales)
        {
            auto ws      = std::ceilf(w * v);
            auto hs      = std::ceilf(h * v);
            auto img0    = opencv::normalize(r);

            auto pnet_in = tensorflow_lite_c_api::make_input_tensor(&m.m_interpreter, 0);
            auto s0      = pnet_in.byte_size();
            auto s1      = opencv::byte_size(img0);

            pnet_in.copy_from_buffer(img0.data, opencv::byte_size(img0));

            auto pnet0 = tensorflow_lite_c_api::make_output_tensor(&m.m_interpreter, 0);
            auto pnet1 = tensorflow_lite_c_api::make_output_tensor(&m.m_interpreter, 1);

            m.m_interpreter.invoke();

            //pnet
            {
                const float t = 0.8f;

                auto pnet0_mat  = mtcnn::make_xtensor_4(pnet0);
                auto pnet1_mat  = mtcnn::make_xtensor_4(pnet1);

                auto imap       = xt::transpose(xt::view(pnet0_mat.m_numpy, 0, xt::all(), xt::all(), 1));
                auto reg        = xt::view(pnet1_mat.m_numpy, 0, xt::all(), xt::all(), xt::all());

                auto dx1        = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 0));
                auto dy1        = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 1));
                auto dx2        = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 2));
                auto dy2        = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 3));

                auto yx         = xt::where(imap > t);
                auto y          = yx[0];
                auto x          = yx[1];

                //auto score      = xt::index_view(imap,  std::make_tuple(y, x) );
                auto score      = xt::index_view(imap, y);


                for (auto i : score)
                {
                    auto f = i;
                    std::cout << f << std::endl;
                }

                
                /*
                auto score      = xt::xarray<float>(filter(imap, yx));

                auto dx1_f      = xt::xarray<float>(filter(dx1, yx));
                auto dy1_f      = xt::xarray<float>(filter(dy1, yx));
                auto dx2_f      = xt::xarray<float>(filter(dx2, yx));
                auto dy2_f      = xt::xarray<float>(filter(dy2, yx));

                reg             = xt::transpose(xt::stack(std::make_tuple(dx1_f, dy1_f, dx2_f, dy2_f), 0));
                */

                //auto b          = test.cbegin();
                //auto e = test.cend();

                //while (b++ != e)
                {
                //    std::cout << *b << std::endl;
                }

                //for (auto i : test)
                {
                    //auto f = i;
                    //std::cout << f << std::endl;
                }
                
                __debugbreak();
                
            }







        }
    }



    return 0;
 }


    