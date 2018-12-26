#pragma once

#include <opencv2/opencv.hpp>

namespace opencv
{
    using mat = cv::Mat;

    mat resize(const mat& r, uint32_t w, uint32_t h)
    {
        mat m;

        cv::resize(r, m, cv::Size(w, h), 0.0, 0.0, cv::INTER_LINEAR);
        return m;
    }

    mat resample(const mat& r, uint32_t h, uint32_t w)
    {
        mat m;

        cv::resize(r, m, cv::Size(w, h), 0.0, 0.0, cv::INTER_AREA);
        return m;
    }

    auto width(const mat& r)
    {
        return r.cols;
    }

    auto height(const mat& r)
    {
        return r.rows;
    }

    auto byte_size(const mat& r)
    {
        return r.u->size;
    }

    auto normalize(const mat& r)
    {
        mat o0;
        r.convertTo(o0, CV_32FC3);

        //convert from 0-255 bytes to floats in the [-1;1]
        mat o1;
        o1 = (o0 - 127.5f) * (1.0f /  128.0f);

        return o1;
    }

    auto to_float(const mat& r)
    {
        mat o0;
        r.convertTo(o0, CV_32FC3);
        return o0;
    }

    auto normalize2(const mat& r)
    {
        //convert from 0-255 bytes to floats in the [-1;1]
        mat o1;
        o1 = (r - 127.5f) * (1.0f / 128.0f);

        return o1;
    }


    mat make_mat(int32_t dims, int32_t* sizes)
    {
        mat m(dims, sizes, CV_32FC1);
        return m;
    }

    mat make_mat(void* data, int32_t dim0, int32_t dim1)
    {
        mat m( dim0, dim1, CV_8UC3, data);
        return m;
    }
}
