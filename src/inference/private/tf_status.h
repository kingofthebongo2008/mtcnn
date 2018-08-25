#pragma once

#include <cstdint>

namespace tensorflow_c_api
{
    class status : private not_copyable
    {

    public:

        enum code : std::uint32_t
        {
            ok = TF_OK,
            canceled = TF_CANCELLED,
            unknown = TF_UNKNOWN,
            invalid_argument = TF_INVALID_ARGUMENT,
            deadline_exceeded = TF_DEADLINE_EXCEEDED,
            not_found = TF_NOT_FOUND,
            already_exists = TF_ALREADY_EXISTS,
            permision_denied = TF_PERMISSION_DENIED,
            unauthenticated = TF_UNAUTHENTICATED,
            resource_exhausted = TF_RESOURCE_EXHAUSTED,
            failed_precondition = TF_FAILED_PRECONDITION,
            aborted = TF_ABORTED,
            out_of_range = TF_OUT_OF_RANGE,
            unimplemented = TF_UNIMPLEMENTED,
            internal = TF_INTERNAL,
            unvailable = TF_UNAVAILABLE,
            data_loss = TF_DATA_LOSS,
        };

    private:
        using status_code = code;

    public:

        status()
        {
            m_status = TF_NewStatus();
        }

        status(TF_Status* m)
        {
            m_status = m;
        }

        status(nullptr_t)
        {
            m_status = nullptr;
        }

        ~status()
        {
            TF_DeleteStatus(m_status);
        }

        status(status&& o)
        {
            m_status = o.m_status;
            o.m_status = nullptr;
        }

        status& operator=(status&& o)
        {
            if (&o != this)
            {
                m_status = o.m_status;
                o.m_status = nullptr;
            }
        }

        operator TF_Status *() const
        {
            return m_status;
        }

        code code() const
        {
            return static_cast<status_code>(TF_GetCode(m_status));
        }

        const char* message() const
        {
            return TF_Message(m_status);
        }

    private:
        TF_Status * m_status;
    };
}
