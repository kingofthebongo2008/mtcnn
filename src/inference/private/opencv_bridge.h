#pragma once

#include <opencv2/opencv.hpp>

namespace opencv
{
    using mat = cv::Mat;

    mat resize(mat r, uint32_t w, uint32_t h)
    {
        mat m;

        cv::resize(r, m, cv::Size(w, h));
        return m;
    }

    auto width(mat r)
    {
        return r.cols;
    }

    auto height(mat r)
    {
        return r.rows;
    }

    auto byte_size(mat r)
    {
        return r.step[0] * r.rows;
    }

    auto normalize(mat r)
    {

        mat o0;
        r.convertTo(o0, CV_32FC3);

        //convert from 0-255 bytes to floats in the [-1;1]
        mat o1;
        o1 = (o0 - 127.5f) * (1.0f / 128.0f);
        return o1;
    }
}
