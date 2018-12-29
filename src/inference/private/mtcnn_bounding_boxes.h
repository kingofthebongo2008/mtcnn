#pragma once

#include <vector>

namespace mtcnn
{
    struct boxes
    {
        std::vector<float> m_x1;
        std::vector<float> m_y1;
        std::vector<float> m_x2;
        std::vector<float> m_y2;

        using value_type = float;

        size_t size() const
        {
            return m_x1.size(); // all vectors match
        }

        bool empty() const
        {
            return size() == 0;
        }
    };
    
    struct boxes_with_score : public boxes
    {
        std::vector<float>    m_score;
    };

    struct bounding_boxes : boxes_with_score
    {
        std::vector<float>    m_score;

        std::vector<float>    m_reg_dx1;
        std::vector<float>    m_reg_dy1;
        std::vector<float>    m_reg_dx2;
        std::vector<float>    m_reg_dy2;

        bool empty()
        {
            return m_score.empty();
        }

        void append(const bounding_boxes& b)
        {
            std::copy(b.m_x1.cbegin(), b.m_x1.cend(), std::back_inserter(m_x1));
            std::copy(b.m_x2.cbegin(), b.m_x2.cend(), std::back_inserter(m_x2));
            std::copy(b.m_y1.cbegin(), b.m_y1.cend(), std::back_inserter(m_y1));
            std::copy(b.m_y2.cbegin(), b.m_y2.cend(), std::back_inserter(m_y2));

            std::copy(b.m_reg_dx1.cbegin(), b.m_reg_dx1.cend(), std::back_inserter(m_reg_dx1));
            std::copy(b.m_reg_dx2.cbegin(), b.m_reg_dx2.cend(), std::back_inserter(m_reg_dx2));
            std::copy(b.m_reg_dy1.cbegin(), b.m_reg_dy1.cend(), std::back_inserter(m_reg_dy1));
            std::copy(b.m_reg_dy2.cbegin(), b.m_reg_dy2.cend(), std::back_inserter(m_reg_dy2));

            std::copy(b.m_score.cbegin(), b.m_score.cend(), std::back_inserter(m_score));
        }
    };

    bounding_boxes make_bounding_boxes(size_t s)
    {
        bounding_boxes r;

        r.m_x1.resize(s);
        r.m_x2.resize(s);

        r.m_y1.resize(s);
        r.m_y2.resize(s);

        r.m_score.resize(s);

        r.m_reg_dx1.resize(s);
        r.m_reg_dy1.resize(s);
        r.m_reg_dx2.resize(s);
        r.m_reg_dy2.resize(s);

        return r;
    }

    boxes_with_score make_boxes_with_score(size_t s)
    {
        boxes_with_score r;

        r.m_x1.resize(s);
        r.m_x2.resize(s);

        r.m_y1.resize(s);
        r.m_y2.resize(s);

        r.m_score.resize(s);

        return r;
    }

    boxes make_boxes(size_t s)
    {
        boxes r;

        r.m_x1.resize(s);
        r.m_x2.resize(s);

        r.m_y1.resize(s);
        r.m_y2.resize(s);


        return r;
    }

    template <typename boxes>
    boxes index_bounding_boxes(const boxes& b, const std::vector<uint16_t> indices)
    {
        boxes r;

        r.m_x1 = index_view(b.m_x1, indices);
        r.m_x2 = index_view(b.m_x2, indices);
        r.m_y1 = index_view(b.m_y1, indices);
        r.m_y2 = index_view(b.m_y2, indices);

        r.m_score = index_view(b.m_score, indices);

        return r;
    }

    template <>
    bounding_boxes index_bounding_boxes<bounding_boxes>(const bounding_boxes& b, const std::vector<uint16_t> indices)
    {
        bounding_boxes r;

        r.m_x1 = index_view(b.m_x1, indices);
        r.m_x2 = index_view(b.m_x2, indices);
        r.m_y1 = index_view(b.m_y1, indices);
        r.m_y2 = index_view(b.m_y2, indices);

        r.m_score = index_view(b.m_score, indices);

        r.m_reg_dx1 = index_view(b.m_reg_dx1, indices);
        r.m_reg_dy1 = index_view(b.m_reg_dy1, indices);
        r.m_reg_dx2 = index_view(b.m_reg_dx2, indices);
        r.m_reg_dy2 = index_view(b.m_reg_dy2, indices);

        return r;
    }


    


}
