#include "pch.h"


#include <memory>

#include "tensorflow_lite_c_api.h"

namespace tensorflow_lite_c_api
{
    struct TFL_Model_Deleter
    {
        void operator()(void* v)
        {
            TFL_DeleteModel( reinterpret_cast<TFL_Model*>(v) );
        }
    };

    using model1 = std::unique_ptr< TFL_Model, TFL_Model_Deleter >;

    struct TFL_InterpreterOptions_Deleter
    {
        void operator()(void* v)
        {
            TFL_DeleteInterpreterOptions(reinterpret_cast<TFL_InterpreterOptions*>(v));
        }
    };

    using interpreter_options = std::unique_ptr< TFL_InterpreterOptions, TFL_InterpreterOptions_Deleter >;

    struct TFL_Interpreter_Deleter
    {
        void operator()(void* v)
        {
            TFL_DeleteInterpreter(reinterpret_cast<TFL_Interpreter*>(v));
        }
    };

    using interpreter = std::unique_ptr< TFL_Interpreter, TFL_Interpreter_Deleter >;
}

int32_t main(int32_t, char**)
{
    tensorflow_lite_c_api::model                m("data/hello_world.tflite");
    tensorflow_lite_c_api::interpreter_options  o(TFL_NewInterpreterOptions());
    TFL_InterpreterOptionsSetNumThreads(o.get(), 4);

    tensorflow_lite_c_api::interpreter          i(TFL_NewInterpreter(m, o.get()));

    auto in                                     = TFL_InterpreterGetInputTensorCount(i.get());
    auto out                                    = TFL_InterpreterGetOutputTensorCount(i.get());

    auto t0                                     = TFL_InterpreterGetInputTensor(i.get(), 0);
    auto t1                                     = TFL_InterpreterGetOutputTensor(i.get(), 0);

    auto s                                      = TFL_InterpreterAllocateTensors(i.get());

    auto t2                                     = TFL_TensorData(t0);
    auto t3                                     = TFL_TensorData(t1);

    auto sz0 = TFL_TensorByteSize(t0);
    auto sz1 = TFL_TensorByteSize(t1);

    float x = 2.0f;
    float y = 0.0f;

    auto s1                                     = TFL_TensorCopyFromBuffer(t0, &x, sizeof(x));
    auto s2                                     = TFL_InterpreterInvoke(i.get());
    auto s3                                     = TFL_TensorCopyToBuffer(t1, &y, sizeof(y));
    
    

    return 0;
}
