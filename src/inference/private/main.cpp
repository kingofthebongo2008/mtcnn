#include "pch.h"

#include <tensorflow/c/c_api.h>
#include <memory>

namespace tensorflow_c_api
{
    class session_options
    {
        public:
        session_options()
        {
            m_options = TF_NewSessionOptions();
        }

        ~session_options()
        {
            TF_DeleteSessionOptions(m_options);
        }

        session_options(const session_options&) = delete;
        session_options& operator=(const session_options&) = delete;

        session_options(session_options&& o)
        {
            m_options = o.m_options;
            o.m_options = nullptr;
        }

        session_options& operator=(session_options&& o)
        {
            if (&o != this)
            {
                m_options = o.m_options;
                o.m_options = nullptr;
            }
        }

        private:

        TF_SessionOptions * m_options;
    };
}



int32_t main(int32_t, char**)
{
    std::unique_ptr<tensorflow_c_api::session_options> m = std::make_unique<tensorflow_c_api::session_options>();

    return 0;
}
