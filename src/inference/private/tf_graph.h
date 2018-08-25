#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class graph : private not_copyable
    {

    public:

        graph()
        {
            m_value = TF_NewGraph();
        }

        graph(TF_Graph* b)
        {
            m_value = b;
        }

        ~graph()
        {
            TF_DeleteGraph(m_value);
        }

        graph(graph&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        graph& operator=(graph&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Graph *() const
        {
            return m_value;
        }

    private:
        TF_Graph * m_value;
    };
}
