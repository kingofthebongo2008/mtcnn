#include "pch.h"

#include <iostream>

#include "opencv_bridge.h"
#include "mtcnn_bridge.h"
#include "mtcnn_numpy.h"
#include "mtcnn_bounding_boxes_compute.h"
#include "mtcnn_nms.h"
#include "mtcnn_rerec.h"
#include "mtcnn_functions.h"
#include "mtcnn_pad.h"
#include "mtcnn_bbreg.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

namespace mtcnn
{
    struct rnet_model : public model
    {
        rnet_model(tensorflow_lite_c_api::model m
            , tensorflow_lite_c_api::interpreter_options  o
            , tensorflow_lite_c_api::interpreter          i) :
            model( 
                std::move(m)
                , std::move(o)
                , std::move(i))
        {

        }



        void resize_input_tensor(uint32_t box_count)
        {
            auto inter = &m_interpreter;
            std::array<int32_t, 4> input_dims = { box_count, 24,24, 3 };
            inter->resize_input_tensor(0, &input_dims[0], input_dims.size());
            inter->allocate_tensors();
        }
    };

    struct onet_model : public model
    {
        onet_model(tensorflow_lite_c_api::model m
            , tensorflow_lite_c_api::interpreter_options  o
            , tensorflow_lite_c_api::interpreter          i) :
            model(
                std::move(m)
                , std::move(o)
                , std::move(i))
        {

        }



        void resize_input_tensor(uint32_t box_count)
        {
            auto inter = &m_interpreter;
            std::array<int32_t, 4> input_dims = { box_count, 48,48, 3 };
            inter->resize_input_tensor(0, &input_dims[0], input_dims.size());
            inter->allocate_tensors();
        }
    };

    std::array< model, 14> make_pnet_models()
    {
        constexpr std::array< const char*, 14> file_names =
        {
            "data/pnet_1600_2560.tflite",
            "data/pnet_960_1536.tflite",
            "data/pnet_672_1076.tflite",
            "data/pnet_471_753.tflite",
            "data/pnet_330_527.tflite",
            "data/pnet_231_369.tflite",
            "data/pnet_162_259.tflite",
            "data/pnet_113_181.tflite",
            "data/pnet_080_127.tflite",
            "data/pnet_056_089.tflite",
            "data/pnet_039_062.tflite",
            "data/pnet_028_044.tflite",
            "data/pnet_019_031.tflite",
            "data/pnet_014_022.tflite"
        };

        return
        {
            mtcnn::make_model<model>(file_names[0]),
            mtcnn::make_model<model>(file_names[1]),
            mtcnn::make_model<model>(file_names[2]),
            mtcnn::make_model<model>(file_names[3]),

            mtcnn::make_model<model>(file_names[4]),
            mtcnn::make_model<model>(file_names[5]),
            mtcnn::make_model<model>(file_names[6]),
            mtcnn::make_model<model>(file_names[7]),

            mtcnn::make_model<model>(file_names[8]),
            mtcnn::make_model<model>(file_names[9]),
            mtcnn::make_model<model>(file_names[10]),
            mtcnn::make_model<model>(file_names[11]),

            mtcnn::make_model<model>(file_names[12]),
            mtcnn::make_model<model>(file_names[13])
        };
    }

    rnet_model make_rnet_model()
    {
        return mtcnn::make_model<rnet_model>("data/rnet.tflite");
    }

    onet_model make_onet_model()
    {
        return mtcnn::make_model<onet_model>("data/onet.tflite");
    }

    struct models_database
    {
        std::array< model, 14> m_pnet_models = make_pnet_models();
        rnet_model             m_rnet_model  = make_rnet_model();
        onet_model             m_onet_model  = make_onet_model();
    };

    models_database make_models_database()
    {
        return models_database();
    }
}

void print_array(const char* file_name, std::vector<uint16_t>& v)
{
    std::ofstream f(file_name, std::ofstream::out);

    for (auto& i : v)
    {
        f << i << "\n";
    }
}

void print_array(const char* file_name, std::vector<float>& v)
{
    std::ofstream f(file_name, std::ofstream::out);

    for (auto& i : v)
    {
        f << std::setprecision(5) << std::fixed << i << "\n";
    }
}


template <typename it>
inline xt::xkeep_slice<std::ptrdiff_t> keep(it begin, it end)
{
    using slice_type = xt::xkeep_slice<std::ptrdiff_t>;
    using container_type = typename slice_type::container_type;
    container_type tmp;
    tmp.resize(end - begin);
    std::copy(begin, end, tmp.begin());
    return slice_type(std::move(tmp));
}

struct result
{
    mtcnn::boxes    m_boxes;
    mtcnn::points   m_points;
};

result detect_face( cv::Mat r, int w, int h)
{
    result res;

    auto scales = mtcnn::make_scales(w, h, mtcnn::minimum_face_size_px, mtcnn::initial_scale);
    auto models = mtcnn::make_models_database();

    auto  img = mtcnn::make_xtensor_2<uint8_t>(r.data, w, h);

    //phase 1

    mtcnn::bounding_boxes total_boxes;

    //Phase 1
    for (auto i = 0U; i < scales.size(); ++i)
    {
        auto v = scales[i];
        auto ws = std::ceilf(w * v);
        auto hs = std::ceilf(h * v);
        auto img0 = opencv::normalize(opencv::resample(r, hs, ws));
        auto inter = &models.m_pnet_models[i].m_interpreter;

        auto pnet_in = tensorflow_lite_c_api::make_input_tensor(inter, 0);
        auto s0 = pnet_in.byte_size();
        auto s1 = opencv::byte_size(img0);

        pnet_in.copy_from_buffer(img0.data, opencv::byte_size(img0));

        auto pnet0 = tensorflow_lite_c_api::make_output_tensor(inter, 0);
        auto pnet1 = tensorflow_lite_c_api::make_output_tensor(inter, 1);

        inter->invoke();

        mtcnn::bounding_boxes boxes = mtcnn::compute_bounding_boxes(pnet0, pnet1, v, 0.8f);

        if (!boxes.empty())
        {
            auto pick = mtcnn::nms(boxes, mtcnn::nms_method::union_value, 0.5f);

            if (!pick.empty())
            {
                total_boxes.append(mtcnn::index_bounding_boxes(boxes, pick));
            }
        }
    }

    if (!total_boxes.empty())
    {
        //get indices that match
        auto pick = mtcnn::nms(total_boxes, mtcnn::nms_method::union_value, 0.7f);

        if (!pick.empty())
        {
            total_boxes = mtcnn::index_bounding_boxes(total_boxes, pick);

            auto regw = mtcnn::sub<uint16_t>(total_boxes.m_x2, total_boxes.m_x1);
            auto regh = mtcnn::sub<uint16_t>(total_boxes.m_y2, total_boxes.m_y1);

            mtcnn::boxes b;

            b.m_x1 = mtcnn::mul<float>(regw, total_boxes.m_reg_dx1);
            b.m_x2 = mtcnn::mul<float>(regw, total_boxes.m_reg_dx2);

            b.m_y1 = mtcnn::mul<float>(regh, total_boxes.m_reg_dy1);
            b.m_y2 = mtcnn::mul<float>(regh, total_boxes.m_reg_dy2);

            auto score = total_boxes.m_score;

            b = mtcnn::add(b, total_boxes);
            b = mtcnn::rerec(b);

            auto b0 = mtcnn::trunc(b);
            auto b1 = mtcnn::pad(b0, w, h);

            auto numbox = b0.size();
            auto tmpimg = mtcnn::make_xtensor_4<float>(numbox, 24, 24, 3);

            for (auto k = 0; k < numbox; ++k)
            {
                auto local_height = b1.m_tmph[k];
                auto local_width = b1.m_tmpw[k];
                auto tmp = mtcnn::make_xtensor_3<uint8_t>(b1.m_tmph[k], b1.m_tmpw[k], 3);
                auto s = tmp.m_numpy.size();
                auto tmpimg_view = xt::view(tmpimg.m_numpy, k, xt::all(), xt::all(), xt::all());

                auto width = b1.m_edx[k] - (b1.m_dx[k] - 1);
                auto height = b1.m_edy[k] - (b1.m_dy[k] - 1);

                for (auto i = 0; i < height; ++i)
                {
                    for (auto j = 0; j < width; ++j)
                    {
                        auto src_y = static_cast<int32_t>(i + b1.m_dy[k] - 1);
                        auto src_x = static_cast<int32_t>(j + b1.m_dx[k] - 1);

                        auto dst_y = static_cast<int32_t>(i + b1.m_y[k] - 1);
                        auto dst_x = static_cast<int32_t>(j + b1.m_x[k] - 1);

                        src_x = std::max(src_x, 0);
                        src_y = std::max(src_y, 0);
                        dst_x = std::max(dst_x, 0);
                        dst_y = std::max(dst_y, 0);
                        tmp.m_numpy[{ src_y, src_x, 0 }] = img[{ dst_y, dst_x, 0 }];
                        tmp.m_numpy[{ src_y, src_x, 1 }] = img[{ dst_y, dst_x, 1 }];
                        tmp.m_numpy[{ src_y, src_x, 2 }] = img[{ dst_y, dst_x, 2 }];
                    }
                }
                std::copy(tmp.m_numpy.cbegin(), tmp.m_numpy.cend(), tmp.m_data.begin());

                opencv::mat m0 = opencv::make_mat(&tmp.m_data[0], local_height, local_width);
                opencv::mat m1 = opencv::resample(opencv::to_float(m0), 24, 24);
                opencv::mat m2 = opencv::normalize2(m1);
                auto        m3 = mtcnn::make_xtensor_3<float>(24, 24, 3, reinterpret_cast<const float*>(m2.data));
                tmpimg_view = m3.m_numpy;
            }

            //phase 2 rnet
            {

                models.m_rnet_model.resize_input_tensor(numbox);

                auto inter = &models.m_rnet_model.m_interpreter;
                auto rnet_in = tensorflow_lite_c_api::make_input_tensor(inter, 0);
                auto s1 = tmpimg.m_numpy.size();

                std::vector<float> buffer;
                buffer.resize(s1);
                std::copy(tmpimg.m_numpy.cbegin(), tmpimg.m_numpy.cend(), buffer.begin());
                rnet_in.copy_from_buffer(&buffer[0], s1 * sizeof(float));

                auto rnet0 = tensorflow_lite_c_api::make_output_tensor(inter, 0);
                auto rnet1 = tensorflow_lite_c_api::make_output_tensor(inter, 1);
                inter->invoke();

                auto t0 = mtcnn::make_xtensor_2(rnet0);
                auto t1 = mtcnn::make_xtensor_2(rnet1);

                t0.m_numpy = xt::transpose(t0.m_numpy);
                t1.m_numpy = xt::transpose(t1.m_numpy);

                auto score = xt::view(t0.m_numpy, 1, xt::all());
                auto ipass = xt::where(score > 0.8f);

                mtcnn::boxes_with_score boxes = mtcnn::make_boxes_with_score(ipass[0].size());

                for (auto i = 0U; i < ipass[0].size(); ++i)
                {
                    auto index = ipass[0][i];

                    boxes.m_x1[i] = b0.m_x1[index];
                    boxes.m_y1[i] = b0.m_y1[index];
                    boxes.m_x2[i] = b0.m_x2[index];
                    boxes.m_y2[i] = b0.m_y2[index];
                    boxes.m_score[i] = score[index];
                }

                if (!boxes.empty())
                {
                    auto mv = xt::view(t1.m_numpy, xt::all(), keep(ipass[0].begin(), ipass[0].end()));
                    pick = mtcnn::nms(boxes, mtcnn::nms_method::union_value, 0.7f);
                    boxes = mtcnn::index_bounding_boxes(boxes, pick);
                    auto mv_picked = xt::view(mv, xt::all(), keep(pick.begin(), pick.end()));
                    boxes = mtcnn::bbreg(boxes, mv_picked);
                    boxes = mtcnn::rerec(boxes);
                }

                //phase 3 onet
                if (!boxes.empty())
                {
                    auto b0 = mtcnn::trunc(boxes);
                    auto b1 = mtcnn::pad(b0, w, h);

                    auto numbox = b0.size();
                    auto tmpimg = mtcnn::make_xtensor_4<float>(numbox, 48, 48, 3);

                    for (auto k = 0; k < numbox; ++k)
                    {
                        auto local_height = b1.m_tmph[k];
                        auto local_width = b1.m_tmpw[k];
                        auto tmp = mtcnn::make_xtensor_3<uint8_t>(b1.m_tmph[k], b1.m_tmpw[k], 3);
                        auto s = tmp.m_numpy.size();
                        auto tmpimg_view = xt::view(tmpimg.m_numpy, k, xt::all(), xt::all(), xt::all());

                        auto width = b1.m_edx[k] - (b1.m_dx[k] - 1);
                        auto height = b1.m_edy[k] - (b1.m_dy[k] - 1);

                        for (auto i = 0; i < height; ++i)
                        {
                            for (auto j = 0; j < width; ++j)
                            {
                                auto src_y = static_cast<int32_t>(i + b1.m_dy[k] - 1);
                                auto src_x = static_cast<int32_t>(j + b1.m_dx[k] - 1);

                                auto dst_y = static_cast<int32_t>(i + b1.m_y[k] - 1);
                                auto dst_x = static_cast<int32_t>(j + b1.m_x[k] - 1);
                                tmp.m_numpy[{ src_y, src_x, 0 }] = img[{ dst_y, dst_x, 0 }];
                                tmp.m_numpy[{ src_y, src_x, 1 }] = img[{ dst_y, dst_x, 1 }];
                                tmp.m_numpy[{ src_y, src_x, 2 }] = img[{ dst_y, dst_x, 2 }];
                            }
                        }

                        std::copy(tmp.m_numpy.cbegin(), tmp.m_numpy.cend(), tmp.m_data.begin());

                        opencv::mat m0 = opencv::make_mat(&tmp.m_data[0], local_height, local_width);
                        opencv::mat m1 = opencv::resample(opencv::to_float(m0), 48, 48);
                        opencv::mat m2 = opencv::normalize2(m1);
                        auto        m3 = mtcnn::make_xtensor_3<float>(48, 48, 3, reinterpret_cast<const float*>(m2.data));
                        tmpimg_view = m3.m_numpy;
                    }

                    {
                        models.m_onet_model.resize_input_tensor(numbox);

                        auto inter = &models.m_onet_model.m_interpreter;
                        auto onet_in = tensorflow_lite_c_api::make_input_tensor(inter, 0);
                        auto s1 = tmpimg.m_numpy.size();

                        std::vector<float> buffer;
                        buffer.resize(s1);
                        std::copy(tmpimg.m_numpy.cbegin(), tmpimg.m_numpy.cend(), buffer.begin());
                        onet_in.copy_from_buffer(&buffer[0], s1 * sizeof(float));

                        auto onet0 = tensorflow_lite_c_api::make_output_tensor(inter, 0);
                        auto onet1 = tensorflow_lite_c_api::make_output_tensor(inter, 1);
                        auto onet2 = tensorflow_lite_c_api::make_output_tensor(inter, 2);

                        inter->invoke();

                        auto t0 = mtcnn::make_xtensor_2(onet0);
                        auto t1 = mtcnn::make_xtensor_2(onet1);
                        auto t2 = mtcnn::make_xtensor_2(onet2);

                        t0.m_numpy = xt::transpose(t0.m_numpy);
                        t1.m_numpy = xt::transpose(t1.m_numpy);

                        auto score = xt::view(t0.m_numpy, 1, xt::all());
                        auto ipass = xt::where(score > 0.8f);

                        mtcnn::boxes_with_score boxes = mtcnn::make_boxes_with_score(ipass[0].size());
                        mtcnn::points            points = mtcnn::make_points(ipass[0].size());

                        for (auto i = 0U; i < ipass[0].size(); ++i)
                        {
                            auto index = ipass[0][i];

                            boxes.m_x1[i] = b0.m_x1[index];
                            boxes.m_y1[i] = b0.m_y1[index];
                            boxes.m_x2[i] = b0.m_x2[index];
                            boxes.m_y2[i] = b0.m_y2[index];
                            boxes.m_score[i] = score[index];
                        }

                        auto w = mtcnn::add<float>(1.0f, mtcnn::sub<float>(boxes.m_x2, boxes.m_x1));
                        auto h = mtcnn::add<float>(1.0f, mtcnn::sub<float>(boxes.m_y2, boxes.m_y1));

                        for (auto i = 0U; i < ipass[0].size(); ++i)
                        {
                            int32_t index = ipass[0][i];

                            points.m_x1[i] = w[i] * (t2.m_numpy[{ index, 0}] + 1.0f) / 2.0f + boxes.m_x1[i] - 1.0f;
                            points.m_x2[i] = w[i] * (t2.m_numpy[{ index, 2}] + 1.0f) / 2.0f + boxes.m_x1[i] - 1.0f;
                            points.m_x3[i] = w[i] * (t2.m_numpy[{ index, 4}] + 1.0f) / 2.0f + boxes.m_x1[i] - 1.0f;
                            points.m_x4[i] = w[i] * (t2.m_numpy[{ index, 6}] + 1.0f) / 2.0f + boxes.m_x1[i] - 1.0f;
                            points.m_x5[i] = w[i] * (t2.m_numpy[{ index, 8}] + 1.0f) / 2.0f + boxes.m_x1[i] - 1.0f;

                            points.m_y1[i] = h[i] * (t2.m_numpy[{ index, 1}] + 1.0f) / 2.0f + boxes.m_y1[i] - 1.0f;
                            points.m_y2[i] = h[i] * (t2.m_numpy[{ index, 3}] + 1.0f) / 2.0f + boxes.m_y1[i] - 1.0f;
                            points.m_y3[i] = h[i] * (t2.m_numpy[{ index, 5}] + 1.0f) / 2.0f + boxes.m_y1[i] - 1.0f;
                            points.m_y4[i] = h[i] * (t2.m_numpy[{ index, 7}] + 1.0f) / 2.0f + boxes.m_y1[i] - 1.0f;
                            points.m_y5[i] = h[i] * (t2.m_numpy[{ index, 9}] + 1.0f) / 2.0f + boxes.m_y1[i] - 1.0f;
                        }

                        if (!boxes.empty())
                        {
                            auto mv = xt::view(t1.m_numpy, xt::all(), keep(ipass[0].begin(), ipass[0].end()));
                            boxes = mtcnn::bbreg(boxes, mv);
                            pick = mtcnn::nms(boxes, mtcnn::nms_method::minimum_value, 0.7f);
                            boxes = mtcnn::index_bounding_boxes(boxes, pick);
                            points = mtcnn::index_bounding_boxes(points, pick);

                            res.m_boxes  = std::move(boxes);
                            res.m_points = std::move(points);
                        }
                    }
                }
            }

        }
    }

    return res;
}


int32_t main(int32_t, char*[])
{ 
    std::cout << "reading data/images/test2.jpg" << std::endl;
    auto r          = cv::imread("data/images/test2.jpg");

    auto w          = opencv::width(r);
    auto h          = opencv::height(r);

    if (w != 2560 || h != 1600)
    {
        std::cerr << "Required width and height are 2560 x 1600" << std::endl;
        return -1;
    }

    auto result = detect_face(r, w, h);

    if (!result.m_points.empty())
    {
        auto&& p = result.m_points;
        auto s   = p.size();
        for (auto i = 0U; i < s; ++i)
        {
            cv::Scalar color(255, 255, 0); 
            cv::circle(r, cv::Point(p.m_x1[i],p.m_y1[i]), 2, color);
            cv::circle(r, cv::Point(p.m_x2[i], p.m_y2[i]), 2, color);
            cv::circle(r, cv::Point(p.m_x3[i], p.m_y3[i]), 2, color);
            cv::circle(r, cv::Point(p.m_x4[i], p.m_y4[i]), 2, color);
            cv::circle(r, cv::Point(p.m_x5[i], p.m_y5[i]), 2, color);
        }

        std::cout << "writing to result2.png" << std::endl;
        cv::imwrite("result2.png", r);

    }

    return 0;
 }


    