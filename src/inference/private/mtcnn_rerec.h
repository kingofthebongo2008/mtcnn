#pragma once

#include <vector>
#include "mtcnn_numpy.h"

namespace mtcnn
{
    template <typename box>
    box rerec(const box& s)
    {
        using value_type = typename box::value_type;

        box r = s;

        auto h      = mtcnn::sub<float>(s.m_y2, s.m_y1);
        auto w      = mtcnn::sub<float>(s.m_x2, s.m_x1);

        auto size   = mtcnn::maximum<float>(w, h);

        auto e0     = mtcnn::mul<float>(0.5f, w);
        auto e1     = mtcnn::mul<float>(0.5f, h);
        auto e2     = mtcnn::mul<float>(0.5f, size);

        auto x1     = mtcnn::add<float>(s.m_x1, mtcnn::sub<float>(e0, e2));
        auto y1     = mtcnn::add<float>(s.m_y1, mtcnn::sub<float>(e1, e2));

        r.m_x1      = std::move(x1);
        r.m_y1      = std::move(y1);

        auto x2     = mtcnn::add<float>(r.m_x1, size);
        auto y2     = mtcnn::add<float>(r.m_y1, size);

        r.m_x2      = std::move(x2);
        r.m_y2      = std::move(y2);
        
        return r;
    }
}
