#pragma once

#include <vector>

#include "mtcnn_bounding_boxes.h"
#include "mtcnn_numpy.h"

namespace mtcnn
{
    enum class nms_method : uint32_t
    {
        minimum_value = 0,
        union_value = 1
    };

    template <typename boxes>
    std::vector<uint16_t> nms(const boxes& s, const nms_method m, const float threshold = 0.8f)
    {
        using v16 = std::vector<uint16_t>;
        using vf  = std::vector<float>;

        v16 sorted_s = argsort(s.m_score);
        v16 pick;
        pick.reserve(s.size());

        auto&& x1 = s.m_x1;
        auto&& y1 = s.m_y1;
        auto&& x2 = s.m_x2;
        auto&& y2 = s.m_y2;

        vf area;

        {
            vf w0 = fold<float>(x1, x2, [](const float x1, const float x2)
            {

                return static_cast<float>(std::max<float>(0.0f, x2 - x1 + 1));
            });

            vf h0 = fold<float>(y1, y2, [](const float y1, const float y2)
            {

                return static_cast<uint16_t>(std::max<int32_t>(0.0f, y2 - y1 + 1));
            });

            area = mul<float>(w0, h0);
        }

        while (!sorted_s.empty())
        {
            auto i = sorted_s.back();

            pick.push_back(i);
            sorted_s.pop_back();

            const auto& idx = sorted_s;

            vf xx1 = maximum<float>(x1[i], index_view(x1, idx));
            vf yy1 = maximum<float>(y1[i], index_view(y1, idx));

            vf xx2 = minimum<float>(x2[i], index_view(x2, idx));
            vf yy2 = minimum<float>(y2[i], index_view(y2, idx));

            vf w = fold<float>(xx1, xx2, [](const float x1, const float x2)
            {
                return static_cast<float>(std::max<float>(0, x2 - x1 + 1));
            });

            vf h = fold<float>(yy1, yy2, [](const float y1, const float y2)
            {
                return static_cast<float>(std::max<float>(0, y2 - y1 + 1));
            });

            vf inter = mul<float>(w, h);

            std::vector<float> o;

            if (m == nms_method::minimum_value)
            {
                o = div<float>(inter, minimum<float>(area[i], index_view(area, idx)));
            }
            else
            {
                float a0 = static_cast<float>(area[i]);

                o = fold<float>(inter, index_view(area, idx), [a0](const float a, const float b)
                {
                    return (static_cast<float>(a) / (a0 + b - a));
                });
            }

            v16 filtered = where_index<uint16_t>(o, [threshold](const float v)
            {
                return v <= threshold;
            });

            sorted_s = index_view(sorted_s, filtered);
        }

        return pick;
    }
}
