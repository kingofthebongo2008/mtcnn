#pragma once

#include "tfl_base.h"
#include "tfl_exception.h"

namespace tensorflow_lite_c_api
{
    class interpreter : private not_copyable
    {

    public:

        interpreter(TFL_Model* m, TFL_InterpreterOptions* o)
        {
            m_value = TFL_NewInterpreter(m, o);
            throw_if_failed_value(m_value);
        }

        ~interpreter()
        {
            if (m_value)
            {
                TFL_DeleteInterpreter(m_value);
            }
        }

        interpreter(interpreter&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        interpreter& operator=(interpreter&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        int32_t get_input_tensor_count() const
        {
            return TFL_InterpreterGetInputTensorCount(m_value);
        }

        TFL_Tensor* get_input_tensor(int32_t index) const
        {
            return TFL_InterpreterGetInputTensor(m_value, index);
        }

        void resize_input_tensor( int32_t input_index, const int* input_dims, int32_t input_dims_size) const
        {
            throw_if_failed(TFL_InterpreterResizeInputTensor(m_value, input_index, input_dims, input_dims_size));
        }

        void allocate_tensors() const
        {
            throw_if_failed(TFL_InterpreterAllocateTensors(m_value));
        }

        void invoke() const
        {
            throw_if_failed(TFL_InterpreterInvoke(m_value));
        }

        int32_t get_output_tensor_count() const
        {
            return TFL_InterpreterGetOutputTensorCount(m_value);
        }

        const TFL_Tensor* get_output_tensor(int32_t index) const
        {
            return TFL_InterpreterGetOutputTensor(m_value, index);
        }

        operator TFL_Interpreter *() const
        {
            return m_value;
        }

    private:
        TFL_Interpreter* m_value = nullptr;
    };
}
