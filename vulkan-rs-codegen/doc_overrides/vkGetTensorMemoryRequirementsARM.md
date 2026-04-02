# Usage Notes

Queries the memory requirements (size, alignment, memory type
bits) for an existing tensor object. Call before
`bind_tensor_memory_arm` to determine the allocation needed.

Requires `VK_ARM_tensors`.
