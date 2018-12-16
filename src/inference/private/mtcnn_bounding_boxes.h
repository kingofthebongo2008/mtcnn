#pragma once

#include <vector>

namespace mtcnn
{
    struct bounding_boxes
    {
        std::vector<uint16_t> m_x1;
        std::vector<uint16_t> m_x2;

        std::vector<uint16_t> m_y1;
        std::vector<uint16_t> m_y2;
        std::vector<float>    m_score;

        std::vector<float>    m_reg_dx1;
        std::vector<float>    m_reg_dy1;
        std::vector<float>    m_reg_dx2;
        std::vector<float>    m_reg_dy2;

        size_t size() const
        {
            return m_score.size(); // all vectors match
        }

        bool empty()
        {
            return m_score.empty();
        }

        void append(const bounding_boxes& b)
        {
            m_x1.insert(m_x1.end(), b.m_x1.cbegin(), m_x1.cend());
            m_x2.insert(m_x2.end(), b.m_x2.cbegin(), m_x2.cend());

            m_y1.insert(m_y1.end(), b.m_y1.cbegin(), m_y1.cend());
            m_y2.insert(m_y2.end(), b.m_y2.cbegin(), m_y2.cend());

            m_score.insert(m_score.end(), b.m_score.cbegin(), m_score.cend());

            m_reg_dx1.insert(m_reg_dx1.end(), b.m_reg_dx1.cbegin(), m_reg_dx1.cend());
            m_reg_dy1.insert(m_reg_dy1.end(), b.m_reg_dy1.cbegin(), m_reg_dy1.cend());
            m_reg_dx2.insert(m_reg_dx2.end(), b.m_reg_dx2.cbegin(), m_reg_dx2.cend());
            m_reg_dy2.insert(m_reg_dy2.end(), b.m_reg_dy1.cbegin(), m_reg_dy2.cend());
        }
    };

    bounding_boxes make_boxes(size_t s)
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


    bounding_boxes index_bounding_boxes(const bounding_boxes& b, const std::vector<uint16_t> indices)
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
