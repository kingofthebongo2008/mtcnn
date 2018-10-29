#include "pch.h"

#include <tensorflow/contrib/lite/experimental/c/c_api.h>



int32_t main(int32_t, char**)
{
    TFL_Model* model = TFL_NewModelFromFile("data/hello_world.tflite");
    
    
    return 0;
}
