#include "pch.h"

#pragma optimize("",off)

#include <iostream>

#include "opencv_bridge.h"
#include "mtcnn_bridge.h"
#include "mtcnn_numpy.h"
#include "mtcnn_bounding_boxes_compute.h"
#include "mtcnn_nms.h"

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
    

    
    //phase 1
    if (true)
    {
        mtcnn::bounding_boxes total_boxes;

        //Phase 1
        for (auto i = 1U; i < scales.size(); ++i)
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
            auto pick = mtcnn::nms(total_boxes, mtcnn::nms_method::union_value, 0.7f);

            if (!pick.empty())
            {
                total_boxes = mtcnn::index_bounding_boxes(total_boxes, pick);



            }
        }
    }



    return 0;
 }


    