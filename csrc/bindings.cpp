#include <torch/extension.h>
#include <iostream>

// Forward declaration of our CUDA function from the .cu file
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);

// Your existing dummy function
torch::Tensor dummy_function(torch::Tensor input_tensor) {
    std::cout << "Hello from C++! Received a tensor of size: " << input_tensor.sizes() << std::endl;
    return input_tensor * 2; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dummy_function", &dummy_function, "A dummy function to test the C++ bridge");
    
    // Add the new CUDA function to Python!
    m.def("matmul", &matmul_cuda, "A naive matrix multiplication in CUDA");
}