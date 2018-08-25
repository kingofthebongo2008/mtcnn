#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class device_list : private not_copyable
    {

    public:

        device_list(TF_DeviceList* b)
        {
            m_value = b;
        }

        ~device_list()
        {
            TF_DeleteDeviceList(m_value);
        }

        device_list(device_list&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        device_list& operator=(device_list&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_DeviceList *() const
        {
            return m_value;
        }

    private:
        TF_DeviceList * m_value;
    };
}
