#pragma once

#include "tfl_interpreter.h"
#include "tfl_tensor.h"

namespace tensorflow_lite_c_api
{
    inline input_tensor make_input_tensor(const interpreter* i, int32_t index)
    {
        return  input_tensor(i->get_input_tensor(index));
    }

    inline output_tensor make_output_tensor(const interpreter* i, int32_t index)
    {
        return  output_tensor(i->get_output_tensor(index));
    }

    inline input_tensor make_input_tensor(const interpreter& i, int32_t index)
    {
        return  input_tensor(i.get_input_tensor(index));
    }

    inline output_tensor make_output_tensor(const interpreter& i, int32_t index)
    {
        return  output_tensor(i.get_output_tensor(index));
    }

}
