#pragma once

#include "tf_base.h"
#include "tf_graph.h"
#include "tf_session_options.h"

namespace tensorflow_c_api
{
    class session : private not_copyable
    {
        public:

        session(const graph* g, const session_options* o)
        {
            status s;
            m_value = TF_NewSession(*g, *o, s);
            throw_if_failed(s);
        }

        session(TF_Session* b)
        {
            m_value = b;
        }

        ~session()
        {
            status s;
            TF_DeleteSession(m_value, s);
        }

        session(session&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        session& operator=(session&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Session *() const
        {
            return m_value;
        }

    private:
        TF_Session * m_value;
    };
}
