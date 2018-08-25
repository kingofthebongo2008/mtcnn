#pragma once

#include "tf_status.h"

namespace tensorflow_c_api
{
    class session_options : private not_copyable
    {
    public:

        session_options()
        {
            m_options = TF_NewSessionOptions();
        }

        ~session_options()
        {
            TF_DeleteSessionOptions(m_options);
        }

        session_options(session_options&& o)
        {
            m_options = o.m_options;
            o.m_options = nullptr;
        }

        session_options& operator=(session_options&& o)
        {
            if (&o != this)
            {
                m_options = o.m_options;
                o.m_options = nullptr;
            }
        }

        void set_target(const char* target)
        {
            TF_SetTarget(m_options, target);
        }

        status set_config(const void* proto, size_t proto_len)
        {
            status r(nullptr);
            TF_SetConfig(m_options, proto, proto_len, r);
            return r;
        }

        operator TF_SessionOptions *() const
        {
            return m_options;
        }

    private:
        TF_SessionOptions * m_options;
    };
}