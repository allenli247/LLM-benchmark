# LLM-Bench-Detective

A custom profiling and benchmarking project aimed at understanding the hardware mechanics of LLM inference using Python and custom CUDA C++ extensions.

## Project Structure

- **`python_runner/`**: Contains the standard Python scripts for running a baseline transformer model (e.g., Gemma-2B). We'll set and trace inference benchmarks, establishing "Tokens per Second" (TPS) and "Time to First Token" (TTFT).
- **`cpp_extension/`**: Contains the source code for building a PyTorch C++/CUDA extension. We will write optimized operators here (such as weight quantization kernels) utilizing memory coalescing, shared memory, and vectorized loads.
- **`analysis/`**: Space for storing profiling outputs and reports from `nsys` (NVIDIA Nsight Systems) and comparing baseline Python implementations with the integrated C++ extension.

## Objective
Targeted at diving deep into model inference profiling rather than model training. The goal is to optimize GPU compute efficiency and track memory bottlenecks directly down to the silicon level.
