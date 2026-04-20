# LLM-Bench-Detective

A custom profiling and benchmarking project aimed at understanding the hardware mechanics of LLM inference using Python and custom CUDA C++ extensions.

## Project Structure

- **`python_runner/`**: Contains the standard Python scripts for running a baseline transformer model (e.g., Gemma-2B). We'll set and trace inference benchmarks, establishing "Tokens per Second" (TPS) and "Time to First Token" (TTFT).
- **`cpp_extension/`**: Contains the source code for building a PyTorch C++/CUDA extension. We will write optimized operators here (such as weight quantization kernels) utilizing memory coalescing, shared memory, and vectorized loads.
- **`analysis/`**: Space for storing profiling outputs and reports from `nsys` (NVIDIA Nsight Systems) and comparing baseline Python implementations with the integrated C++ extension.

llm-inference-engine/
│
├── README.md                   # Document your TPS, TTFT, and hardware findings here
├── requirements.txt            # torch, transformers, accelerate, etc.
├── setup.py                    # CRITICAL: The script that compiles your C++/CUDA code
│
├── python_runner/                     # All your high-level Python code
│   ├── baseline_runner.py      # The Hugging Face script (Part 1)
│   ├── custom_runner.py        # The script that imports and tests your C++ kernel (Part 2)
│   └── utils.py                # Helper functions (e.g., measuring TPS/TTFT)
│
├── cpp_extension/                       # "C Source" - All your low-level code
│   ├── bindings.cpp            # The PyBind11 code that connects C++ to Python
│   ├── matmul_kernel.cu        # The actual CUDA code running on the GPU
│   └── matmul_kernel.h         # C++ header files defining your functions
│
└── analysis/                  # Where you will save your GPU analysis
    └── nsys_reports/           # Store the .nsys-rep files generated in Part 3

## Objective
Targeted at diving deep into model inference profiling rather than model training. The goal is to optimize GPU compute efficiency and track memory bottlenecks directly down to the silicon level.
