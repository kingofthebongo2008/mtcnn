#include "pch.h"

#include "tensorflow_c_api.h"

namespace tensorflow {

    /// Tag for the `gpu` graph.
    constexpr char kSavedModelTagGpu[] = "gpu";

    /// Tag for the `tpu` graph.
    constexpr char kSavedModelTagTpu[] = "tpu";

    /// Tag for the `serving` graph.
    constexpr char kSavedModelTagServe[] = "serve";

    /// Tag for the `training` graph.
    constexpr char kSavedModelTagTrain[] = "train";

}  // namespace tensorflow


int32_t main(int32_t, char**)
{
    try
    {
        using namespace tensorflow_c_api;
        using namespace std;

        session_options options;
        graph           g;
        buffer          meta_graph;
        buffer          b;

        const char* tags[] = { tensorflow::kSavedModelTagTrain };

        auto session = load_session_from_saved_model(options, b, "./all_in_one", tags, 1, g, meta_graph);
    }
    catch (const tensorflow_c_api::exception& e)
    {
        __debugbreak();
    }
    
    return 0;
}
