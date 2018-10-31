#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class model : private not_copyable
    {

    public:

        model(const void* d, size_t s)
        {
            m_value = TFL_NewModel(d, s);
            throw_if_failed_value(m_value);
        }

        model(const char* file_name)
        {
            m_value = TFL_NewModelFromFile(file_name);
            throw_if_failed_value(m_value);
        }

        ~model()
        {
            if (m_value)
            {
                TFL_DeleteModel(m_value);
            }
        }

        model(model&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        model& operator=(model&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TFL_Model *() const
        {
            return m_value;
        }

    private:
        TFL_Model* m_value = nullptr;
    };
}
