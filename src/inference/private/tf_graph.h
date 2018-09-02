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

        void set_tensor_shape(output o, const std::int64_t* dims, const std::int32_t num_dims )
        {
            status s;
            TF_GraphSetTensorShape(m_value, o, dims, num_dims, s);
            throw_if_failed(s);
        }

        std::int32_t get_tensor_num_dims(output o) const
        {
            status s;
            auto result = TF_GraphGetTensorNumDims(m_value, o, s);
            throw_if_failed(s);
            return result;
        }

        std::int32_t get_tensor_shape(output o, std::int64_t* dims, std::int32_t num_dims) const
        {
            status s;
            TF_GraphGetTensorShape(m_value, o, dims, num_dims, s);
            throw_if_failed(s);
        }

        operator TF_Graph *() const
        {
            return m_value;
        }

    private:
        TF_Graph * m_value;
    };
}
