#pragma once

#include <vector>
#include <algorithm>

namespace mtcnn
{
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
        return fold<uint16_t>(c, v, [](const uint16_t a, const uint16_t b)
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
}
