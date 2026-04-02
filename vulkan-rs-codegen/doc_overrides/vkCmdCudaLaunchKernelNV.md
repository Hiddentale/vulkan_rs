# Usage Notes

Launches a CUDA kernel from within a Vulkan command buffer.
The kernel is specified by a CUDA function handle created with
`create_cuda_function_nv`. Grid dimensions and parameters are
provided in the launch info.

Requires `VK_NV_cuda_kernel_launch`.
