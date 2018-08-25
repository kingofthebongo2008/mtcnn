#pragma once

#include "tf_base.h"
#include "tf_status.h"

#include <exception>
#include <string>

namespace tensorflow_c_api
{
    class exception : public std::exception
    {

    public:

    exception(status::error_code c, const char* m) : m_code(c), m_message(m)
    {

    }

        
    const char* what() const override
    {
        return m_message.c_str();
    }

    private:

        status::error_code m_code;
        std::string m_message;
        
    };


    inline void throw_if_failed(const status* s)
    {
        auto c = s->code();

        if (c != status::ok)
        {
            throw exception(c, s->message());
        }
    }

    inline void throw_if_failed(const status& s)
    {
        auto c = s.code();

        if (c != status::ok)
        {
            throw exception(c, s.message());
        }
    }
}
