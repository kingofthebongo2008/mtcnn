#pragma once

#include <cstdint>
#include <memory>

#include <tensorflow/c/c_api.h>

namespace tensorflow_c_api
{
    class not_copyable
    {
        public:
        not_copyable() = default;
        ~not_copyable() = default;

        not_copyable(const not_copyable&) = delete;
        not_copyable& operator=(const not_copyable&) = delete;

        not_copyable(not_copyable&&) = default;
        not_copyable& operator=(not_copyable&&) = default;
    };

    enum class data : std::uint32_t
    {
        float_t         = TF_FLOAT,
        double_t        = TF_DOUBLE,
        int32_t         = TF_INT32,
        uint8_t         = TF_UINT8,
        int16_t         = TF_INT16,
        int8_t          = TF_INT8,
        string_t        = TF_STRING,
        complex64_t     = TF_COMPLEX64,     // Single-precision complex
        complex_t       = TF_COMPLEX,       // Old identifier kept for API backwards compatibility
        int64_t         = TF_INT64,
        bool_t          = TF_BOOL,
        qint8_t         = TF_QINT8,         // Quantized int8
        quint8_t        = TF_QUINT8,        // Quantized uint8
        qint32_t        = TF_QINT32,        // Quantized int32
        bloat16_t       = TF_BFLOAT16,      // Float32 truncated to 16 bits.  Only for cast ops.
        qint16_t        = TF_QINT16,        // Quantized int16
        quint16_t       = TF_QUINT16,       // Quantized uint16
        uint16_t        = TF_UINT16,
        complex128_t    = TF_COMPLEX128,    // Double-precision complex
        half_t          = TF_HALF,
        resource_t      = TF_RESOURCE,
        variant_t       = TF_VARIANT,
        uint32_t        = TF_UINT32,
        uint64_t        = TF_UINT64,
    };

    struct operation
    {
        TF_Operation* m_o;

        operator TF_Operation*() const
        {
            return m_o;
        }
    };

    struct output
    {
        operation       m_o;
        ::std::int32_t  m_index;

        operator TF_Output () const
        {
            return TF_Output{ m_o, m_index };
        }
    };

    struct input
    {
        operation       m_o;
        ::std::int32_t  m_index;

        operator TF_Output () const
        {
            return TF_Output{ m_o, m_index };
        }
    };

}
