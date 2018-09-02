#pragma once

#include "tf_base.h"
#include "tf_graph.h"
#include "tf_session_options.h"
#include "tf_session.h"

namespace tensorflow_c_api
{
    inline session load_session_from_saved_model( const TF_SessionOptions* o, const TF_Buffer* b, const  char* export_dir, const char* const* tags, int32_t tags_len, TF_Graph* g, TF_Buffer* meta_graph_def)
    {
        status s;

        TF_Session* r = TF_LoadSessionFromSavedModel(o, b, export_dir, tags, tags_len, g, meta_graph_def, s);
        throw_if_failed(s);
        return session(r);
    }

}
