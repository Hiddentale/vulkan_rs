# Usage Notes

Binds device memory to one or more tensors. Each bind info
specifies the tensor, memory, and offset. Must be called before
the tensor is used in any command. Similar to
`bind_buffer_memory2` / `bind_image_memory2`.

Requires `VK_ARM_tensors`.
