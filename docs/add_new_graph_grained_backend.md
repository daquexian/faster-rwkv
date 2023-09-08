## Add a New Graph-Grained Backend

Assume that your backend is named ABC,

1. Add a new device enum value kABC in tensor.h.

2. Create kernels/ABC/init_model.cpp, register an init_model function. In this function you need to do neccessary initialization and assign model->_extra to any data specific to your backend so that you can retrieve the data afterwards. Example: kernels/ncnn/init_model.cpp

3. Create kernels/ABC/model_forward.cpp. Following the comments in kernels/ncnn/model_forward.cpp.
