#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class operation_description : private not_copyable
    {

    public:

        operation_description() : m_value(nullptr)
        {

        }

        
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
        
        void add_input(TF_Output input)
        {
            TF_AddInput(m_value, input);
        }

        void add_input_list(const TF_Output* inputs, std::int32_t num_inputs )
        {
            TF_AddInputList(m_value, reinterpret_cast<const TF_Output*>(inputs), num_inputs);
        }

        void add_control_input(TF_Operation* input)
        {
            TF_AddControlInput(m_value, input);
        }

        void set_attr_string(const char* attr_name, const void* value, size_t length)
        {
            TF_SetAttrString(m_value, attr_name, value, length);
        }

        void set_attr_string_list(const char* attr_name, const void* const* values, const size_t* lengths, int32_t num_values)
        {
            TF_SetAttrStringList(m_value, attr_name, values, lengths, num_values);
        }

        void set_attr_int(const char* attr_name, int64_t value)
        {
            TF_SetAttrInt(m_value, attr_name, value);
        }

        void set_attr_int_list(const char* attr_name, const int64_t* values, int32_t num_values)
        {
            TF_SetAttrIntList(m_value, attr_name, values, num_values);
        }

        void set_attr_float(const char* attr_name, float value)
        {
            TF_SetAttrFloat(m_value, attr_name, value);
        }

        void set_attr_int_list(const char* attr_name, const float* values, int32_t num_values)
        {
            TF_SetAttrFloatList(m_value, attr_name, values, num_values);
        }

        void set_attr_bool(const char* attr_name, uint8_t value)
        {
            TF_SetAttrBool(m_value, attr_name, value);
        }

        void set_attr_bool_list(const char* attr_name, const uint8_t* values, int32_t num_values)
        {
            TF_SetAttrBoolList(m_value, attr_name, values, num_values);
        }

        void set_attr_type(const char* attr_name, data type)
        {
            TF_SetAttrType(m_value, attr_name, static_cast<TF_DataType>(type));
        }

        void set_attr_type_list(const char* attr_name, const data* values, int32_t num_values)
        {
            TF_SetAttrTypeList(m_value, attr_name, reinterpret_cast<const TF_DataType*>(values), num_values);
        }

        void set_attr_func_name(const char* attr_name, const char* value, size_t length)
        {
            TF_SetAttrFuncName(m_value, attr_name, value, length);
        }

        void set_attr_shape(const char* attr_name, const int64_t* dims, int32_t num_dims)
        {
            TF_SetAttrShape(m_value, attr_name, dims, num_dims);
        }

        void set_attr_shape_list(const char* attr_name, const int64_t* const* values, const int32_t* lengths, int32_t num_values)
        {
            TF_SetAttrShapeList(m_value, attr_name, values, lengths, num_values);
        }

        void set_attr_tensor_shape_proto(const char* attr_name, const void* proto, size_t proto_len)
        {
            status s;
            TF_SetAttrTensorShapeProto(m_value, attr_name, proto, proto_len, s);
            throw_if_failed(s);
        }

        void set_attr_tensor_shape_proto_list(const char* attr_name, const void* const* values, const size_t* lengths, int32_t num_values)
        {
            status s;
            TF_SetAttrTensorShapeProtoList(m_value, attr_name, values, lengths, num_values, s);
            throw_if_failed(s);
        }

        void set_attr_value_proto(const char* attr_name, const void* proto, size_t proto_len)
        {
            status s;
            TF_SetAttrValueProto(m_value, attr_name, proto, proto_len, s);
            throw_if_failed(s);
        }

        void set_attr_tensor(const char* attr_name, TF_Tensor* value)
        {
            status s;
            TF_SetAttrTensor(m_value, attr_name, value, s);
        }

        void set_attr_tensor_list(const char* attr_name, TF_Tensor * const * values, int32_t num_values)
        {
            status s;
            TF_SetAttrTensorList(m_value, attr_name, values, num_values, s);
        }



        /*

        TF_CAPI_EXPORT extern void TF_SetAttrTensor(TF_OperationDescription* desc,
            const char* attr_name,
            TF_Tensor* value,
            TF_Status* status);
        TF_CAPI_EXPORT extern void TF_SetAttrTensorList(TF_OperationDescription* desc,
            const char* attr_name,
            TF_Tensor* const* values,
            int num_values,
            TF_Status* status);
            TF_Status* status);
        */

    private:

        TF_OperationDescription* m_value;
    };
}
