#define __CUDA_NO_FP4_CONVERSIONS__
#define __CUDA_NO_FP6_CONVERSIONS__
#include <torch/extension.h>
#include <cuda_runtime.h>

// 1. THE GPU KERNEL: This runs on thousands of GPU threads simultaneously
__global__ void matmul_cuda_kernel(
    const float* a, const float* b, float* c, int N) {
    
    // Find out which thread this is (its x/y coordinate on the GPU)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the thread is within the matrix boundaries, do the math!
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

// 2. THE HOST FUNCTION: This runs on the CPU and launches the GPU kernel
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    // Ensure the tensors are actually on the GPU and are contiguous in memory
    TORCH_CHECK(a.device().is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor B must be on CUDA");
    
    // Get the size of the square matrix (N x N)
    int N = a.size(0);

    // Create an empty tensor on the GPU to hold the result
    auto c = torch::zeros_like(a);

    // Group the GPU threads into 16x16 blocks
    dim3 threadsPerBlock(16, 16);
    // Calculate how many blocks we need to cover the whole matrix
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // LAUNCH THE KERNEL!
    matmul_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        N
    );

    return c;
}