#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class interpreter_options : private not_copyable
    {

    public:

        interpreter_options()
        {
            m_value = TFL_NewInterpreterOptions();
            throw_if_failed_value(m_value);
        }

        ~interpreter_options()
        {
            if (m_value)
            {
                TFL_DeleteInterpreterOptions(m_value);
            }
        }

        interpreter_options(interpreter_options&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        interpreter_options& operator=(interpreter_options&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        void set_num_threads(int32_t threads)
        {
            TFL_InterpreterOptionsSetNumThreads(m_value, threads);
        }
        
        operator TFL_InterpreterOptions *() const
        {
            return m_value;
        }

    private:
        TFL_InterpreterOptions* m_value = nullptr;
    };
}
