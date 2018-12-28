#pragma once

#include "mtcnn_numpy.h"
#include "mtcnn_bounding_boxes.h"

namespace mtcnn
{
    struct padded_boxes
    {
        std::vector<float> m_dy;
        std::vector<float> m_edy;

        std::vector<float> m_dx;
        std::vector<float> m_edx;

        std::vector<float>   m_y;
        std::vector<float>   m_ey;

        std::vector<float>   m_x;
        std::vector<float>   m_ex;

        std::vector<float> m_tmpw;
        std::vector<float> m_tmph;
    };

    template <typename a>
    padded_boxes pad(const a& s, int32_t w, int32_t h)
    {
        auto tmpw = add<float>(1, sub<float>(s.m_x2, s.m_x1));
        auto tmph = add<float>(1, sub<float>(s.m_y2, s.m_y1));

        auto numbox = s.size();

        auto edx    = tmpw;
        auto edy    = tmph;

        auto dx     = ones<float>(numbox);
        auto dy     = ones<float>(numbox);

        auto x      = s.m_x1;
        auto y      = s.m_y1;
        auto ex     = s.m_x2;
        auto ey     = s.m_y2;


        for (auto i = 0U; i < ex.size(); ++i)
        {
            if (ex[i] > w)
            {
                edx[i]  = w - ex[i] + tmpw[i]; //( w - x2 + x2 - x1 )  -> map from (x2-x1) to (w-x1)
                ex[i]   = w;
            }
        }

        for (auto i = 0U; i < ex.size(); ++i)
        {
            if (ex[i] < 1)
            {
                dx[i] = 2 - x[i];
                x[i]  = 1;
            }
        }

        for (auto i = 0U; i < ey.size(); ++i)
        {
            if (ey[i] > h)
            {
                edy[i] = h - ey[i] + tmph[i]; //( w - y2 + y2 - x1 )  -> map from (y2-y1) to (h-y1)
                ey[i] = h;
            }
        }

        for (auto i = 0U; i < ey.size(); ++i)
        {
            if (ey[i] < 1)
            {
                dy[i] = 2 - y[i];
                y[i]  = 1;
            }
        }

        padded_boxes r;

        r.m_dy = std::move(dy);
        r.m_edy = std::move(edy);

        r.m_dx = std::move(dx);
        r.m_edx = std::move(edx);

        r.m_y   = std::move(y);
        r.m_ey  = std::move(ey);

        r.m_x   = std::move(x);
        r.m_ex  = std::move(ex);

        r.m_tmpw = std::move(tmpw);
        r.m_tmph = std::move(tmph);


        return r;
    }
}
