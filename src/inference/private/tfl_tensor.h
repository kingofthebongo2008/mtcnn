#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{

    class input_tensor
    {
        public:

        input_tensor() = default;

        input_tensor(TFL_Tensor* m) : m_value(m)
        {
        }

        ~input_tensor() = default;
        input_tensor(input_tensor&& o) = default;
        input_tensor& operator=(input_tensor&& o) = default;


        void copy_from_buffer(const void* input_data, size_t input_data_size)
        {
            throw_if_failed(TFL_TensorCopyFromBuffer(m_value, input_data, input_data_size));

        }

        void copy_to_buffer(void* output_data, size_t output_data_size)
        {
            throw_if_failed(TFL_TensorCopyToBuffer(m_value, output_data, output_data_size));
        }

        size_t byte_size() const
        {
            return TFL_TensorByteSize(m_value);
        }

        int32_t num_dims() const
        {
            return TFL_TensorNumDims(m_value);
        }

        int32_t dim(int32_t dim_index) const
        {
            return TFL_TensorDim(m_value, dim_index);
        }

        TFL_Type type() const
        {
            return TFL_TensorType(m_value);
        }

    private:

        TFL_Tensor * m_value = nullptr;
    };

    class output_tensor
    {

    public:

        output_tensor() = default;

        output_tensor(const TFL_Tensor* m) : m_value(m)
        {

        }

        ~output_tensor() = default;
        output_tensor(output_tensor&& o) = default;
        output_tensor& operator=(output_tensor&& o) = default;

        void copy_to_buffer(void* output_data, size_t output_data_size)
        {
            throw_if_failed(TFL_TensorCopyToBuffer(m_value, output_data, output_data_size));
        }

        size_t byte_size() const
        {
            return TFL_TensorByteSize(m_value);
        }

        int32_t num_dims() const
        {
            return TFL_TensorNumDims(m_value);
        }

        int32_t dim(int32_t dim_index) const
        {
            return TFL_TensorDim(m_value, dim_index);
        }

        TFL_Type type() const
        {
            return TFL_TensorType(m_value);
        }

    private:

        const TFL_Tensor * m_value = nullptr;
    };
}
