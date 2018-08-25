#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class tensor : private not_copyable
    {

    public:

        tensor(data d, const std::int64_t* dims, std::int32_t num_dims, size_t len)
        {
            m_value = TF_AllocateTensor(static_cast<TF_DataType>(d), dims, num_dims, len);
        }

        tensor(TF_Tensor* b)
        {
            m_value = b;
        }

        ~tensor()
        {
            TF_DeleteTensor(m_value);
        }

        tensor(tensor&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        tensor& operator=(tensor&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Tensor *() const
        {
            return m_value;
        }

        data type()  const
        {
            return static_cast<data>(TF_TensorType(m_value));
        }

        std::int32_t num_dims()  const
        {
            return TF_NumDims(m_value);
        }

        size_t byte_size()  const
        {
            return TF_TensorByteSize(m_value);
        }

        void* bytes()  const
        {
            return TF_TensorData(m_value);
        }

    private:
        TF_Tensor * m_value;
    };
}
