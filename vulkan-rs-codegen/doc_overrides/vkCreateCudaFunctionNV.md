# Usage Notes

Creates a CUDA function handle from a CUDA module, identifying
a specific kernel entry point. Use with
`cmd_cuda_launch_kernel_nv` to dispatch the kernel.

Destroy with `destroy_cuda_function_nv`.

Requires `VK_NV_cuda_kernel_launch`.
