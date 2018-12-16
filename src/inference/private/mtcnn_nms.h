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

    std::vector<uint16_t> nms(const bounding_boxes& s, const nms_method m, const float threshold = 0.8f)
    {
        using v16 = std::vector<uint16_t>;
        v16 sorted_s = argsort(s.m_score);
        v16 pick;
        pick.reserve(s.size());

        auto&& x1 = s.m_x1;
        auto&& y1 = s.m_y1;
        auto&& x2 = s.m_x2;
        auto&& y2 = s.m_y2;

        v16 area;

        {
            v16 w0 = fold<uint16_t>(x1, x2, [](const uint16_t x1, const uint16_t x2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, x2 - x1 + 1));
            });

            v16 h0 = fold<uint16_t>(y1, y2, [](const uint16_t y1, const uint16_t y2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, y2 - y1 + 1));
            });

            area = mul(w0, h0);
        }

        while (!sorted_s.empty())
        {
            auto i = sorted_s.back();

            pick.push_back(i);
            sorted_s.pop_back();

            const auto& idx = sorted_s;

            v16 xx1 = maximum(x1[i], index_view(x1, idx));
            v16 yy1 = maximum(y1[i], index_view(y1, idx));

            v16 xx2 = minimum(x2[i], index_view(x2, idx));
            v16 yy2 = minimum(y2[i], index_view(y2, idx));

            v16 w = fold<uint16_t>(xx1, xx2, [](const uint16_t x1, const uint16_t x2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, x2 - x1 + 1));
            });

            v16 h = fold<uint16_t>(yy1, yy2, [](const uint16_t y1, const uint16_t y2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, y2 - y1 + 1));
            });

            v16 inter = mul(w, h);
            std::vector<float> o;

            if (m == nms_method::minimum_value)
            {
                o = div(inter, minimum(area[i], index_view(area, idx)));
            }
            else
            {
                float a0 = static_cast<float>(area[i]);

                o = fold<float>(inter, index_view(area, idx), [a0](const uint16_t a, const uint16_t b)
                {
                    return (static_cast<float>(a) / (a0 + b - a));
                });
            }

            v16 filtered = where_index(o, [threshold](const float v)
            {
                return v <= threshold;
            });

            sorted_s = index_view(sorted_s, filtered);
        }

        return pick;
    }
}
