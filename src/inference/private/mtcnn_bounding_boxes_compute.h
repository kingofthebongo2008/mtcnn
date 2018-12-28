#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

#include "mtcnn_bounding_boxes.h"

#include "tensorflow_lite_c_api.h"

namespace mtcnn
{
    template <typename t>
    struct numpy_wrapper
    {
        std::vector<t>  m_data;
        xt::xarray<t>   m_numpy;
    };

    template <typename t>
    auto make_xtensor_4(size_t dim0, size_t dim1, size_t dim2, size_t dim3)
    {
        std::array<size_t, 4> shape = { dim0, dim1, dim2, dim3 };
        auto size                   = dim0 * dim1 * dim2 * dim3;

        numpy_wrapper<t> w;
        w.m_data.resize(size);
        w.m_numpy = xt::adapt(&w.m_data[0], size, xt::no_ownership(), shape);
        return w;
    }

    template <typename t>
    auto make_xtensor_3(size_t dim0, size_t dim1, size_t dim2)
    {
        std::array<size_t, 3> shape = { dim0, dim1, dim2 };
        auto size                   = dim0 * dim1 * dim2;

        numpy_wrapper<t> w;
        w.m_data.resize(size);
        w.m_numpy = xt::adapt(&w.m_data[0], size, xt::no_ownership(), shape);
        return w;
    }

    template <typename t>
    auto make_xtensor_3(size_t dim0, size_t dim1, size_t dim2, const t* data)
    {
        std::array<size_t, 3> shape = { dim0, dim1, dim2 };
        auto size = dim0 * dim1 * dim2;

        numpy_wrapper<t> w;
        w.m_data.resize(size);
        w.m_numpy = xt::adapt(data, size, xt::no_ownership(), shape);
        return w;
    }

    auto make_xtensor_4(const tensorflow_lite_c_api::output_tensor& tensor)
    {
        std::array<size_t, 4> shape = { static_cast<size_t>(tensor.dim(0)), static_cast<size_t>(tensor.dim(1)), static_cast<size_t>(tensor.dim(2)), static_cast<size_t>(tensor.dim(3)) };
        auto size                   = tensor.byte_size() / sizeof(float);

        numpy_wrapper<float> w;
        w.m_data.resize(size);

        tensor.copy_to_buffer(&w.m_data[0], tensor.byte_size());
        w.m_numpy = xt::adapt(&w.m_data[0], size, xt::no_ownership(), shape, xt::layout_type::row_major);

        return w;
    }

    auto make_xtensor_2(const tensorflow_lite_c_api::output_tensor& tensor)
    {
        std::array<size_t, 2> shape = { static_cast<size_t>(tensor.dim(0)), static_cast<size_t>(tensor.dim(1)) };
        auto size = tensor.byte_size() / sizeof(float);

        numpy_wrapper<float> w;
        w.m_data.resize(size);

        tensor.copy_to_buffer(&w.m_data[0], tensor.byte_size());
        w.m_numpy = xt::adapt(&w.m_data[0], size, xt::no_ownership(), shape, xt::layout_type::row_major);

        return w;
    }

    template<typename t>
    xt::xarray<t> make_xtensor_2(t* data, uint32_t w, uint32_t h)
    {
        std::array<size_t, 3> shape = { static_cast<size_t>(h), static_cast<size_t>(w), 3 };

        return  xt::adapt(data, w * h * 3 * sizeof(t), xt::no_ownership(), shape, xt::layout_type::row_major);
    }

    xt::xarray<float> make_xtensor_2(float* data, uint32_t w, uint32_t h)
    {
        std::array<size_t, 3> shape = { static_cast<size_t>(h), static_cast<size_t>(w), 3 };
        return  xt::adapt(data, w * h * 3, xt::no_ownership(), shape, xt::layout_type::row_major);
    }

    bounding_boxes compute_bounding_boxes(const tensorflow_lite_c_api::output_tensor& pnet0, const tensorflow_lite_c_api::output_tensor& pnet1, const float scale = 1.0f, const float threshold = 0.8f)
    {
        auto pnet0_mat = mtcnn::make_xtensor_4(pnet0);
        auto pnet1_mat = mtcnn::make_xtensor_4(pnet1);

        auto imap = xt::transpose(xt::view(pnet0_mat.m_numpy, 0, xt::all(), xt::all(), 1));
        auto reg = xt::view(pnet1_mat.m_numpy, 0, xt::all(), xt::all(), xt::all());

        auto dx1 = xt::transpose(xt::view(reg, xt::all(), xt::all(), 0));
        auto dy1 = xt::transpose(xt::view(reg, xt::all(), xt::all(), 1));
        auto dx2 = xt::transpose(xt::view(reg, xt::all(), xt::all(), 2));
        auto dy2 = xt::transpose(xt::view(reg, xt::all(), xt::all(), 3));

        const float t = threshold;
        const float s = scale;
        const auto  stride = 2.0;
        const auto  cell_size = 12.0;

        auto yx = xt::where(imap >= t); //filter
        xt::xarray<size_t> y = xt::adapt(yx[1]);
        xt::xarray<size_t> x = xt::adapt(yx[0]);

        auto b = y.size();
        auto boxes = make_bounding_boxes(b);

        //suitable for concurrent execution

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = imap[{x[i], y[i]}];
                boxes.m_score[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t>(std::trunc((y[i] * stride + 1.0) / scale));
                boxes.m_y1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t>(std::trunc((x[i] * stride + 1.0) / scale));
                boxes.m_x1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t>(std::trunc((x[i] * stride + cell_size) / scale));
                boxes.m_x2[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t> (std::trunc((y[i] * stride + cell_size) / scale));
                boxes.m_y2[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dx1[{x[i], y[i]}];
                boxes.m_reg_dx1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dy1[{x[i], y[i]}];
                boxes.m_reg_dy1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dx2[{x[i], y[i]}];
                boxes.m_reg_dx2[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dy2[{x[i], y[i]}];
                boxes.m_reg_dy2[i] = value;
            }
        }
        return boxes;
    }
}
