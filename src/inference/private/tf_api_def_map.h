#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class api_def_map : private not_copyable
    {

    public:

        api_def_map(TF_ApiDefMap* b)
        {
            m_value = b;
        }

        ~api_def_map()
        {
            TF_DeleteApiDefMap(m_value);
        }

        api_def_map(api_def_map&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        api_def_map& operator=(api_def_map&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_ApiDefMap *() const
        {
            return m_value;
        }

    private:
        TF_ApiDefMap * m_value;
    };
}
