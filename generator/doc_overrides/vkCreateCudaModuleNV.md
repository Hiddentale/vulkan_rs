# Usage Notes

Creates a CUDA module from PTX or cubin data. The module
contains one or more kernel entry points that can be extracted
with `create_cuda_function_nv`.

Destroy with `destroy_cuda_module_nv`.

Requires `VK_NV_cuda_kernel_launch`.
