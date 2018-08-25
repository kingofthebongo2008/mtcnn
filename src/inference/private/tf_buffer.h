#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class buffer : private not_copyable
    {
    public:
        buffer()
        {
            m_value = TF_NewBuffer();
        }

        buffer(TF_Buffer* b)
        {
            m_value = b;
        }

        buffer(const void* proto, size_t proto_len)
        {
            m_value = TF_NewBufferFromString(proto, proto_len);
        }

        ~buffer()
        {
            TF_DeleteBuffer(m_value);
        }

        buffer(buffer&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        buffer& operator=(buffer&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Buffer *() const
        {
            return m_value;
        }

    private:
        TF_Buffer * m_value;
    };
}
