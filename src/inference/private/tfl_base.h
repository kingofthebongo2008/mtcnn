#pragma once

#include <cstdint>
#include <memory>

#include <tensorflow/contrib/lite/experimental/c/c_api.h>

namespace tensorflow_lite_c_api
{
    class not_copyable
    {
        public:
        not_copyable() = default;
        ~not_copyable() = default;

        not_copyable(const not_copyable&) = delete;
        not_copyable& operator=(const not_copyable&) = delete;

        not_copyable(not_copyable&&) = default;
        not_copyable& operator=(not_copyable&&) = default;
    };
}
