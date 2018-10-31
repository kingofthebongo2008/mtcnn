#pragma once

#include "tfl_base.h"


#include <exception>
#include <string>

namespace tensorflow_lite_c_api
{
    class exception : public std::exception
    {

    public:

    exception(const TFL_Status  s) : m_status(s)
    {

    }

    const char* what() const override
    {
        return "tensorflow lite exception";
    }

    private:
        TFL_Status m_status;
        
    };

    inline void throw_if_failed( const TFL_Status s )
    {
        if (s != kTfLiteOk)
        {
            throw exception(s);
        }
    }

    template <typename T>
    inline void throw_if_failed_value(const T* v)
    {
        if (v == nullptr)
        {
            throw exception(kTfLiteError);
        }
    }
}
