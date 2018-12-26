#include "pch.h"

#pragma optimize("",off)

#include <iostream>

#include "opencv_bridge.h"
#include "mtcnn_bridge.h"
#include "mtcnn_numpy.h"
#include "mtcnn_bounding_boxes_compute.h"
#include "mtcnn_nms.h"
#include "mtcnn_rerec.h"
#include "mtcnn_functions.h"
#include "mtcnn_pad.h"

namespace mtcnn
{
    std::array< model, 14> make_models()
    {
        constexpr std::array< const char*, 14> file_names =
        {
            "data/pnet_1600_2560.tflite",
            "data/pnet_960_1536.tflite",
            "data/pnet_672_1076.tflite",
            "data/pnet_471_753.tflite",
            "data/pnet_330_527.tflite",
            "data/pnet_231_369.tflite",
            "data/pnet_162_259.tflite",
            "data/pnet_113_181.tflite",
            "data/pnet_080_127.tflite",
            "data/pnet_056_089.tflite",
            "data/pnet_039_062.tflite",
            "data/pnet_028_044.tflite",
            "data/pnet_019_031.tflite",
            "data/pnet_014_022.tflite"
        };

        return
        {
            mtcnn::make_model(file_names[0]),
            mtcnn::make_model(file_names[1]),
            mtcnn::make_model(file_names[2]),
            mtcnn::make_model(file_names[3]),

            mtcnn::make_model(file_names[4]),
            mtcnn::make_model(file_names[5]),
            mtcnn::make_model(file_names[6]),
            mtcnn::make_model(file_names[7]),

            mtcnn::make_model(file_names[8]),
            mtcnn::make_model(file_names[9]),
            mtcnn::make_model(file_names[10]),
            mtcnn::make_model(file_names[11]),

            mtcnn::make_model(file_names[12]),
            mtcnn::make_model(file_names[13])
        };
    }

    struct models_database
    {
        std::array< model, 14> m_models = make_models();
    };

    models_database make_models_database()
    {
        return models_database();
    }
}

void print_array(const char* file_name, std::vector<uint16_t>& v)
{
    std::ofstream f(file_name, std::ofstream::out);

    for (auto& i : v)
    {
        f << i << "\n";
    }
}

void print_array(const char* file_name, std::vector<float>& v)
{
    std::ofstream f(file_name, std::ofstream::out);

    for (auto& i : v)
    {
        f << std::setprecision(5) << std::fixed << i << "\n";
    }
}

/*
{
    auto shape = m3.m_numpy.shape();
    auto s0 = shape[0];
    auto s1 = shape[1];

    std::ofstream f("img.txt", std::ofstream::out);

    for (auto i = 0; i < s0; ++i)
    {
        for (auto j = 0; j < s1; ++j)
        {
            auto    v0 = m3.m_numpy[{ i, j, 0}];
            auto    v1 = m3.m_numpy[{ i, j, 1}];
            auto    v2 = m3.m_numpy[{ i, j, 2}];
            f << v0 << ",\n" << v1 << ",\n" << v2 << ",\n";
        }
    }
    __debugbreak();
}
*/


int32_t main(int32_t, char*[])
{ 
    auto r          = cv::imread("data/images/test1.jpg");

    auto w          = opencv::width(r);
    auto h          = opencv::height(r);

    if (w != 2560 || h != 1600)
    {
        std::cerr << "Required width and height are 2560 x 1600" << std::endl;
        return -1;
    }

    auto scales     = mtcnn::make_scales(w, h, mtcnn::minimum_face_size_px, mtcnn::initial_scale);
    auto models     = mtcnn::make_models_database();

    auto  m0        = mtcnn::make_model("data/mtcnn.tflite");
    auto  img       = mtcnn::make_xtensor_2<uint8_t>(r.data, w, h);
    
    //phase 1
    if (true)
    {
        mtcnn::bounding_boxes total_boxes;

        //Phase 1
        for (auto i = 8U; i < scales.size(); ++i)
        {
            auto v       = scales[i];
            auto ws      = std::ceilf(w * v);
            auto hs      = std::ceilf(h * v);
            auto img0    = opencv::normalize(opencv::resample(r, hs, ws));
            auto inter   = &models.m_models[i].m_interpreter;

            auto pnet_in = tensorflow_lite_c_api::make_input_tensor(inter, 0);
            auto s0      = pnet_in.byte_size();
            auto s1      = opencv::byte_size(img0);

            pnet_in.copy_from_buffer(img0.data, opencv::byte_size(img0));

            auto pnet0    = tensorflow_lite_c_api::make_output_tensor(inter, 0);
            auto pnet1    = tensorflow_lite_c_api::make_output_tensor(inter, 1);

            inter->invoke();

            mtcnn::bounding_boxes boxes = mtcnn::compute_bounding_boxes(pnet0, pnet1, v, 0.8f);

            if (!boxes.empty())
            {
                auto pick = mtcnn::nms(boxes, mtcnn::nms_method::union_value, 0.5f);

                if (!pick.empty())
                {
                    total_boxes.append(mtcnn::index_bounding_boxes(boxes, pick));
                }
            }
        }

        if (!total_boxes.empty())
        {
            //get indices that match
            auto pick = mtcnn::nms(total_boxes, mtcnn::nms_method::union_value, 0.7f);

            if (!pick.empty())
            {
                total_boxes = mtcnn::index_bounding_boxes(total_boxes, pick);

                auto regw   = mtcnn::sub<uint16_t>(total_boxes.m_x2, total_boxes.m_x1);
                auto regh   = mtcnn::sub<uint16_t>(total_boxes.m_y2, total_boxes.m_y1);

                mtcnn::boxes<float > b;

                b.m_x1      = mtcnn::mul<float>(regw, total_boxes.m_reg_dx1);
                b.m_x2      = mtcnn::mul<float>(regw, total_boxes.m_reg_dx2);

                b.m_y1      = mtcnn::mul<float>(regh, total_boxes.m_reg_dy1);
                b.m_y2      = mtcnn::mul<float>(regh, total_boxes.m_reg_dy2);

                auto score  = total_boxes.m_score;

                b = mtcnn::add<float>(b, total_boxes);
                b = mtcnn::rerec(b);

                auto b0     = mtcnn::trunc<int32_t>(b);
                auto b1     = mtcnn::pad(b0, w, h);

                auto numbox = b0.size();
                auto tmpimg = mtcnn::make_xtensor_4<float>(numbox, 24, 24, 3);

                for (auto k = 0; k < numbox; ++k)
                {
                    auto local_height   = b1.m_tmph[k];
                    auto local_width    = b1.m_tmpw[k];
                    auto tmp            = mtcnn::make_xtensor_3<uint8_t>(b1.m_tmph[k], b1.m_tmpw[k], 3);
                    auto s              = tmp.m_numpy.size();
                    auto tmpimg_view    = xt::view(tmpimg.m_numpy, k, xt::all(), xt::all(), xt::all() );

                    auto width          = b1.m_edx[k] - (b1.m_dx[k] - 1);
                    auto height         = b1.m_edy[k] - (b1.m_dy[k] - 1);

                    for (auto i = 0; i < height; ++i)
                    {
                        for (auto j = 0; j < width; ++j)
                        {
                            auto src_y = i + b1.m_dy[k] - 1;
                            auto src_x = j + b1.m_dx[k] - 1;

                            auto dst_y = i + b1.m_y[k] - 1;
                            auto dst_x = j + b1.m_x[k] - 1;
                            tmp.m_numpy[{ src_y, src_x, 0 }] = img[{ dst_y, dst_x, 0 }];
                            tmp.m_numpy[{ src_y, src_x, 1 }] = img[{ dst_y, dst_x, 1 }];
                            tmp.m_numpy[{ src_y, src_x, 2 }] = img[{ dst_y, dst_x, 2 }];
                        }
                    }
                    std::copy(tmp.m_numpy.cbegin(), tmp.m_numpy.cend(), tmp.m_data.begin());

                    opencv::mat m0 = opencv::make_mat(&tmp.m_data[0], local_height, local_width);
                    opencv::mat m1 = opencv::resample(opencv::to_float(m0), 24, 24);
                    opencv::mat m2 = opencv::normalize2(m1);
                    auto        m3 = mtcnn::make_xtensor_3<float>(24, 24, 3, reinterpret_cast<const float*>(m2.data));
                    tmpimg_view = m3.m_numpy;
                }


            }
        }
    }
    
    return 0;
 }


    