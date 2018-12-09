#pragma once

#include <vector>
#include <algorithm>

#include "tensorflow_lite_c_api.h"

namespace mtcnn
{
    const float threshold_pnet = 0.8f;
    const float threshold_onet = 0.8f;
    const float threshold_rnet = 0.8f;

    const float minimum_face_size_px = 20;
    const float initial_scale = 0.7f;

    inline std::vector< double > make_scales(uint32_t width, uint32_t height, float min_face_size_px, float factor)
    {
        auto minl = std::min<double>(static_cast<double>(width), static_cast<double>(height));
        auto m = 12.0f / std::max<double>(min_face_size_px, 12.0f);

        //create scale pyramid
        std::vector<double> scales;

        auto factor_count = 0.0f;

        scales.push_back(1.0f);

        minl = minl * m;
        while (minl >= 12.0f)
        {
            auto scale = m * std::pow(factor, factor_count);
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

    inline model make_model(const char* model_file)
    {
        tensorflow_lite_c_api::model                m(model_file);
        tensorflow_lite_c_api::interpreter_options  o;

        o.set_num_threads(8);
        tensorflow_lite_c_api::interpreter          i(m, o);
        i.allocate_tensors();
        return { std::move(m), std::move(o), std::move(i) };
    }

}
