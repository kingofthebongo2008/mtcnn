#pragma once

#pragma optimize("",off)
#include <vector>

#include "mtcnn_bounding_boxes.h"
#include "mtcnn_numpy.h"

namespace mtcnn
{
    template <typename box, typename reg>
    box bbreg(const box& b, const reg& reg)
    {
        auto w = add<float>(1.0f, sub<float>(b.m_x2, b.m_x1));
        auto h = add<float>(1.0f, sub<float>(b.m_y2, b.m_y1));

        box r = b;

        auto s = b.size();
        r.m_x1.resize(s);
        r.m_y1.resize(s);
        r.m_x2.resize(s);
        r.m_y2.resize(s);

        for (auto i = 0; i < s; ++i)
        {
            auto t = b.m_x1[i] + w[i] * reg[{0,i}];
            r.m_x1[i] = t;
        }

        for (auto i = 0; i < s; ++i)
        {
            auto t = b.m_y1[i] + h[i] * reg[{1, i}];
            r.m_y1[i] = t;
        }

        for (auto i = 0; i < s; ++i)
        {
            auto t = b.m_x2[i] + w[i] * reg[{2, i}];
            r.m_x2[i] = t;
        }

        for (auto i = 0; i < s; ++i)
        {
            auto t = b.m_y2[i] + h[i] * reg[{3, i}];
            r.m_y2[i] = t;
        }

        return r;
    }
  
}
