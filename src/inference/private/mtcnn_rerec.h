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

        auto h      = sub<float>(s.m_y2, s.m_y1);
        auto w      = sub<float>(s.m_x2, s.m_x1);

        auto size   = maximum<float>(w, h);

        auto e0     = mul<float>(0.5f, w);
        auto e1     = mul<float>(0.5f, h);
        auto e2     = mul<float>(0.5f, size);

        auto x1     = add<float>(s.m_x1, sub<float>(e0, e2));
        auto y1     = add<float>(s.m_y1, sub<float>(e1, e2));

        r.m_x1      = std::move(x1);
        r.m_y1      = std::move(y1);

        auto x2     = add<float>(r.m_x1, size);
        auto y2     = add<float>(r.m_y1, size);

        r.m_x2      = std::move(x2);
        r.m_y2      = std::move(y2);
        
        return r;
    }
}
