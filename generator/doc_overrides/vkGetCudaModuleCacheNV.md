# Usage Notes

Retrieves the compiled cache data from a CUDA module for
serialization. Call once with a null buffer to query the size,
then again with an appropriately sized buffer. Feed the data
back into `create_cuda_module_nv` on the next run to skip
compilation.

Requires `VK_NV_cuda_kernel_launch`.
