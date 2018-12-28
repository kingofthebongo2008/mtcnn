#pragma once

#include "mtcnn_bounding_boxes.h"
#include "mtcnn_numpy.h"

namespace mtcnn
{
    template <typename a, typename b> boxes add(const a& v0, const b& v1)
    {
        boxes result;

        result.m_x1 = mtcnn::add<float>(v0.m_x1, v1.m_x1);
        result.m_y1 = mtcnn::add<float>(v0.m_y1, v1.m_y1);
        result.m_x2 = mtcnn::add<float>(v0.m_x2, v1.m_x2);
        result.m_y2 = mtcnn::add<float>(v0.m_y2, v1.m_y2);

        return result;
    }

    template <typename a> boxes trunc(const a& v0)
    {
        boxes result;

        result.m_x1 = mtcnn::trunc<float>(v0.m_x1);
        result.m_y1 = mtcnn::trunc<float>(v0.m_y1);
        result.m_x2 = mtcnn::trunc<float>(v0.m_x2);
        result.m_y2 = mtcnn::trunc<float>(v0.m_y2);

        return result;
    }

}
