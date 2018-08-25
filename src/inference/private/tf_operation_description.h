#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class operation_description : private not_copyable
    {

    public:
        operation_description(TF_Graph* b)
        {
            m_value = TF_NewOperation(b, "test", "name");
        }

        ~operation_description()
        {

        }

        operation_description(operation_description&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        operation_description& operator=(operation_description&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_OperationDescription *() const
        {
            return m_value;
        }

    private:
        TF_OperationDescription * m_value;
    };
}
