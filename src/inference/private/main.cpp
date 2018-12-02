#include "pch.h"

#pragma optimize("",off)

#include <iostream>

#include "opencv_bridge.h"
#include "mtcnn_bridge.h"

namespace
{
    void test_hello_world()
    {
        using namespace tensorflow_lite_c_api;

        model                m("data/hello_world.tflite");
        interpreter_options  o;

        o.set_num_threads(8);

        interpreter          i(m, o);

        auto in = i.get_input_tensor_count();
        auto out = i.get_output_tensor_count();

        auto t0 = input_tensor(i.get_input_tensor(0));
        auto t1 = output_tensor(i.get_output_tensor(0));

        i.allocate_tensors();

        float x = 2.0f;
        float y = 0.0f;

        t0.copy_from_buffer(&x, sizeof(x));
        i.invoke();
        t1.copy_to_buffer(&y, sizeof(y));
        std::cout << "Result is " << y << std::endl;
    }
}

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace mtcnn
{
    struct numpy_wrapper
    {
        std::vector<float>  m_data;
        xt::xarray<float>   m_numpy;
    };

    auto make_xtensor_4(const tensorflow_lite_c_api::output_tensor& tensor)
    {
        std::array<size_t, 4> shape = { static_cast<size_t>( tensor.dim(0)), static_cast<size_t>(tensor.dim(1)), static_cast<size_t>(tensor.dim(2)), static_cast<size_t>(tensor.dim(3)) };
        auto size_in_floats         = tensor.byte_size() / sizeof(float);

        numpy_wrapper w;
        w.m_data.resize(size_in_floats);

        tensor.copy_to_buffer(&w.m_data[0], tensor.byte_size() );
        w.m_numpy = xt::adapt(w.m_data, shape);

        return w;
    }

    struct bounding_boxes
    {
        std::vector<uint16_t> m_x0;
        std::vector<uint16_t> m_x1;

        std::vector<uint16_t> m_y0;
        std::vector<uint16_t> m_y1;
        std::vector<float>    m_score;

        std::vector<float>    m_reg_dx1;
        std::vector<float>    m_reg_dy1;
        std::vector<float>    m_reg_dx2;
        std::vector<float>    m_reg_dy2;

        size_t size() const
        {
            return m_score.size(); // all vectors match
        }
    };

    bounding_boxes make_boxes( size_t s )
    {
        bounding_boxes r;

        r.m_x0.resize(s);
        r.m_x1.resize(s);

        r.m_y0.resize(s);
        r.m_y1.resize(s);

        r.m_score.resize(s);

        r.m_reg_dx1.resize(s);
        r.m_reg_dy1.resize(s);
        r.m_reg_dx2.resize(s);
        r.m_reg_dy2.resize(s);

        return r;
    }

    bounding_boxes compute_bounding_boxes( const tensorflow_lite_c_api::output_tensor& pnet0, const tensorflow_lite_c_api::output_tensor& pnet1, const float threshold = 0.8f, const float scale = 1.0f)
    {
        auto pnet0_mat = mtcnn::make_xtensor_4(pnet0);
        auto pnet1_mat = mtcnn::make_xtensor_4(pnet1);

        auto imap = xt::transpose(xt::view(pnet0_mat.m_numpy, 0, xt::all(), xt::all(), 1));
        auto reg = xt::view(pnet1_mat.m_numpy, 0, xt::all(), xt::all(), xt::all());

        auto dx1 = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 0));
        auto dy1 = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 1));
        auto dx2 = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 2));
        auto dy2 = xt::transpose(xt::view(reg, xt::all(), xt::all(), xt::all(), 3));

        const float t           = threshold;
        const float s           = scale;
        const auto  stride      = 2;
        const auto  cell_size   = 12;
        
        auto yx = xt::where(imap > t); //filter
        xt::xarray<size_t> y = xt::adapt(yx[0]);
        xt::xarray<size_t> x = xt::adapt(yx[1]);

        auto b = y.size();
        auto boxes = make_boxes(b);

        //suitable for concurrent execution

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = imap[{y[i], x[i]}];
                boxes.m_score[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t>(std::roundf((y[i] * stride + 1) / scale));
                boxes.m_y0[i] = value;
            }
        }

        {
            //todo: y1 can be deduced from y0
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t> (std::roundf((y[i] * stride + cell_size - 1) / scale));
                boxes.m_y1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = static_cast<uint16_t>(std::roundf((x[i] * stride + 1) / scale));
                boxes.m_x0[i] = value;
            }
        }


        {
        //todo: x1 can be deduced from x0
        for (auto i = 0; i < b; ++i)
        {
            auto value = static_cast<uint16_t>(std::roundf((x[i] * stride + cell_size - 1) / scale));
            boxes.m_x1[i] = value;
        }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dx1[{y[i], x[i]}];
                boxes.m_reg_dx1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dy1[{y[i], x[i]}];
                boxes.m_reg_dy1[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dx2[{y[i], x[i]}];
                boxes.m_reg_dx2[i] = value;
            }
        }

        {
            for (auto i = 0; i < b; ++i)
            {
                auto value = dy2[{y[i], x[i]}];
                boxes.m_reg_dy2[i] = value;
            }
        }
        return boxes;
    }

    std::vector<uint16_t> argsort(const std::vector<float>& score)
    {
        std::vector<uint16_t> sorted_s;

        sorted_s.resize(score.size());

        std::generate(sorted_s.begin(), sorted_s.end(),
            []()
        {
            static uint16_t i = 0;
            return i++;
        });

        std::sort(sorted_s.begin(), sorted_s.end(), [&score](auto&& a, auto&& b)
        {
            return score[a] < score[b];
        });

        return sorted_s;
    }

    template <typename r, typename op >
    std::vector<r> fold(const uint16_t c, const std::vector<uint16_t>& v, op o)
    {
        std::vector<r> res(v.size());

        for (auto i = 0U; i < v.size(); ++i)
        {
            res[i] = (o(c, v[i]));
        }
        return res;
    }

    template <typename r, typename op >
    std::vector<r> fold(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1, op o)
    {
        std::vector<r>  res(v0.size());

        for (auto i = 0U; i < v0.size(); ++i)
        {
            res[i] = (o(v0[i], v1[i]));
        }
        return res;
    }

    template <typename op >
    std::vector<uint16_t> where_index(const std::vector<float>& v0, op o)
    {
        std::vector<uint16_t>  res;
        res.reserve(v0.size());

        for (auto i = 0U; i < v0.size(); ++i)
        {
            if (o(v0[i]))
            {
                res.push_back(i);
            }
        }
        return res;
    }

    std::vector<uint16_t> maximum(const uint16_t c, const std::vector<uint16_t>& v)
    {
        return fold<uint16_t> (c, v, [](const uint16_t a, const uint16_t b)
        {
            return static_cast<uint16_t> (std::max(a, b));
        });
    }

    std::vector<uint16_t> maximum(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1)
    {
        return fold<uint16_t>(v0, v1, [](const uint16_t a, const uint16_t b)
        {
            return static_cast<uint16_t> (std::max(a, b));
        });
    }

    std::vector<uint16_t> minimum(const uint16_t c, const std::vector<uint16_t>& v)
    {
        return fold<uint16_t>(c, v, [](const uint16_t a, const uint16_t b)
        {
            return (std::min(a, b));
        });
    }

    std::vector<uint16_t> minimum(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1)
    {
        return fold<uint16_t>(v0, v1, [](const uint16_t a, const uint16_t b)
        {
            return static_cast<uint16_t>(std::min(a, b));
        });
    }

    std::vector<uint16_t> mul(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1)
    {
        return fold<uint16_t>(v0, v1, [](const uint16_t a, const uint16_t b)
        {
            return static_cast<uint16_t>(a) * static_cast<uint16_t>(b);
        });
    }

    std::vector<float> div(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1)
    {
        return fold<float>(v0, v1, [](const uint16_t a, const uint16_t b)
        {
            return static_cast<float>(a) / static_cast<float>(b);
        });
    }

    template <typename r>    std::vector<r> index_view(const std::vector<r>& values, const std::vector<uint16_t>& indices)
    {
        std::vector<r> res(indices.size());

        for (auto i = 0U; i < indices.size(); ++i)
        {
            res[i] = values[indices[i]];
        }
        return res;
    }

    enum class nms_method : uint32_t
    {
        minimum_value = 0,
        union_value   = 1
    };

    
    std::vector<uint16_t> nms( const bounding_boxes& s, const nms_method m, const float threshold = 0.8f )
    {
        using v16    = std::vector<uint16_t>;
        v16 sorted_s = argsort(s.m_score);
        v16 pick;
        pick.reserve(s.size());

        auto&& x1 = s.m_x0;
        auto&& y1 = s.m_y0;
        auto&& x2 = s.m_x1;
        auto&& y2 = s.m_y1;

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

            area = mul( w0, h0 );
        }
        
        while (!sorted_s.empty())
        {
            auto i  = sorted_s.back();

            pick.push_back(i);
            sorted_s.pop_back();

            const auto& idx = sorted_s;

            v16 xx1 = maximum(x1[i], index_view(x1, idx));
            v16 yy1 = maximum(y1[i], index_view(y1, idx));

            v16 xx2 = minimum(x2[i], index_view(x2, idx));
            v16 yy2 = minimum(y2[i], index_view(y2, idx));

            v16 w   = fold<uint16_t>(xx1, xx2, [] ( const uint16_t x1, const uint16_t x2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, x2 - x1 + 1));
            });

            v16 h  = fold<uint16_t>(yy1, yy2, [](const uint16_t y1, const uint16_t y2)
            {
                return static_cast<uint16_t>(std::max<int32_t>(0, y2 - y1 + 1));
            });

            v16 inter = mul( w, h );
            std::vector<float> o;

            if ( m == nms_method::minimum_value)
            {
                o = div( inter, minimum(area[i], index_view(area, idx)) );
            }
            else
            {
                uint16_t a0 = static_cast<float>(area[i]);

                o = fold<float>(inter, index_view(area, idx), [a0](const uint16_t a, const uint16_t b)
                {
                    return ( static_cast<float>( a ) / (a0 + b - a));
                });
            }

            v16 filtered = where_index( o, [threshold](const float v)
            {
                return v < threshold;
            });

            sorted_s = index_view(sorted_s, filtered);

        }
    
        return pick;
    }
}

int32_t main(int32_t, char*[])
{ 
    auto r          = cv::imread("data/images/test1.jpg");

    auto w          = opencv::width(r);
    auto h          = opencv::height(r);
    auto scales     = mtcnn::make_scales(w, h, mtcnn::minimum_face_size_px, mtcnn::initial_scale);
    auto m          = mtcnn::make_model("data/mtcnn.tflite");

    m.m_interpreter.allocate_tensors();
    auto ot = m.m_interpreter.get_output_tensor_count();


    if (true)
    {
        for (auto& v : scales)
        {
            auto ws      = std::ceilf(w * v);
            auto hs      = std::ceilf(h * v);
            auto img0    = opencv::normalize(r);

            auto pnet_in = tensorflow_lite_c_api::make_input_tensor(&m.m_interpreter, 0);
            auto s0      = pnet_in.byte_size();
            auto s1      = opencv::byte_size(img0);

            pnet_in.copy_from_buffer(img0.data, opencv::byte_size(img0));

            auto pnet0 = tensorflow_lite_c_api::make_output_tensor(&m.m_interpreter, 0);
            auto pnet1 = tensorflow_lite_c_api::make_output_tensor(&m.m_interpreter, 1);

            m.m_interpreter.invoke();

            mtcnn::bounding_boxes boxes = mtcnn::compute_bounding_boxes(pnet0, pnet1);
            auto                  pick = mtcnn::nms(boxes, mtcnn::nms_method::union_value, 0.5f);

            __debugbreak();
                
        }
    }



    return 0;
 }


    