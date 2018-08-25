#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class library : private not_copyable
    {

    public:

        library(TF_Library* b)
        {
            m_value = b;
        }

        ~library()
        {
            TF_DeleteLibraryHandle(m_value);
        }

        library(library&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        library& operator=(library&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Library *() const
        {
            return m_value;
        }

    private:
        TF_Library * m_value;
    };
}
