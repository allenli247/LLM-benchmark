from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_kernels',
    ext_modules=[
        CUDAExtension(                 # <--- Must be CUDAExtension
            name='custom_kernels',
            sources=[
                'csrc/bindings.cpp', 
                'csrc/matmul_kernel.cu' # <--- Must be included here!
            ],      
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)