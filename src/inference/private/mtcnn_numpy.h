#pragma once

#include <vector>
#include <algorithm>

namespace mtcnn
{
    std::vector<uint16_t> argsort(const std::vector<float>& score)
    {
        std::vector<uint16_t> sorted_s;

        sorted_s.resize(score.size());

        uint16_t v = 0;

        std::generate(sorted_s.begin(), sorted_s.end(),
            [&v]()
        {
            return v++;
        });

        std::sort(sorted_s.begin(), sorted_s.end(), [&score](auto&& a, auto&& b)
        {
            return score[a] < score[b];
        });

        return sorted_s;
    }

    template <typename r, typename a, typename op >
    std::vector<r> fold(const uint16_t c, const std::vector<a>& v, op o)
    {
        std::vector<r> res(v.size());

        for (auto i = 0U; i < v.size(); ++i)
        {
            res[i] = (o(c, v[i]));
        }
        return res;
    }

    template <typename r, typename a, typename b, typename op >
    std::vector<r> fold(const std::vector<a>& v0, const std::vector<b>& v1, op o)
    {
        std::vector<r>  res(v0.size());

        for (auto i = 0U; i < v0.size(); ++i)
        {
            res[i] = (o(v0[i], v1[i]));
        }
        return res;
    }

    template <typename r, typename a, typename op >
    std::vector<r> where_index(const std::vector<a>& v0, op o)
    {
        std::vector<r>  res;
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

    template <typename r, typename a>
    std::vector<r> maximum(const uint16_t c, const std::vector<a>& v)
    {
        return fold<uint16_t>(c, v, [](const uint16_t x, const a y)
        {
            return static_cast<uint16_t> (std::max(x, y));
        });
    }


    template <typename r, typename a, typename b>
    std::vector<r> maximum(const std::vector<a>& v0, const std::vector<b>& v1)
    {
        return fold<r>(v0, v1, [](const a x, const b y)
        {
            return static_cast<r> (std::max(x, y));
        });
    }

    template <typename r, typename a, typename b>
    std::vector<r> minimum(const a c, const std::vector<b>& v)
    {
        return fold<r>(c, v, [](const a x, const b y)
        {
            return (std::min(x, y));
        });
    }

    template <typename r, typename a, typename b>
    std::vector<r> minimum(const std::vector<uint16_t>& v0, const std::vector<uint16_t>& v1)
    {
        return fold<r>(v0, v1, [](const a x, const b y)
        {
            return static_cast<uint16_t>(std::min(x, y));
        });
    }

    template <typename r, typename a, typename b>
    std::vector<r> mul(const std::vector<a>& v0, const std::vector<b>& v1)
    {
        return fold<r>(v0, v1, [](const a x, const b y)
        {
            return static_cast<r>(x * y);
        });
    }

    template <typename r, typename a, typename b>
    std::vector<r> div(const std::vector<a>& v0, const std::vector<b>& v1)
    {
        return fold<r>(v0, v1, [](const a x, const b y)
        {
            return static_cast<r>(x) / static_cast<r>(y);
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
}
