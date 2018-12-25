#pragma once

#include "mtcnn_bounding_boxes.h"
#include "mtcnn_numpy.h"

namespace mtcnn
{
    template <typename r, typename a, typename b> boxes<r> add(const a& v0, const b& v1)
    {
        boxes<r> result;

        result.m_x1 = mtcnn::add<r>(v0.m_x1, v1.m_x1);
        result.m_y1 = mtcnn::add<r>(v0.m_y1, v1.m_y1);
        result.m_x2 = mtcnn::add<r>(v0.m_x2, v1.m_x2);
        result.m_y2 = mtcnn::add<r>(v0.m_y2, v1.m_y2);

        return result;
    }

    template <typename r, typename a> boxes<r> trunc(const a& v0)
    {
        boxes<r> result;

        result.m_x1 = mtcnn::trunc<r>(v0.m_x1);
        result.m_y1 = mtcnn::trunc<r>(v0.m_y1);
        result.m_x2 = mtcnn::trunc<r>(v0.m_x2);
        result.m_y2 = mtcnn::trunc<r>(v0.m_y2);

        return result;
    }

}
