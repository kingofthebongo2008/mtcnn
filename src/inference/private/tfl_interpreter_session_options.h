#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class interpreter_session_options : private not_copyable
    {

    public:

        interpreter_session_options(const void* d, size_t s)
        {
            m_value = TFL_Newinterpreter_session_options(d, s);
            throw_if_failed_value(m_value);
        }

        interpreter_session_options(const char* file_name)
        {
            m_value = TFL_Newinterpreter_session_optionsFromFile(file_name);
            throw_if_failed_value(m_value);
        }

        ~interpreter_session_options()
        {
            if (m_value)
            {
                TFL_Deleteinterpreter_session_options(m_value);
            }
        }

        interpreter_session_options(interpreter_session_options&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        interpreter_session_options& operator=(interpreter_session_options&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TFL_interpreter_session_options *() const
        {
            return m_value;
        }

    private:
        TFL_interpreter_session_options* m_value = nullptr;
    };
}
