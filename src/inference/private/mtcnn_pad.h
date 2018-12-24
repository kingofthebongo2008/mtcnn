#pragma once

#include "mtcnn_numpy.h"

namespace mtcnn
{
    template <typename box>
    box rerec(const box& s)
    {
        box r = s;
        return r;
    }
}
