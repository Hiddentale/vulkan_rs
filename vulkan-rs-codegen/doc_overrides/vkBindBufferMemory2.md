# Usage Notes

Binds memory to one or more buffers in a single call. This is the
Vulkan 1.1 batch version of `bind_buffer_memory`.

Each `BindBufferMemoryInfo` specifies a buffer, memory object, and
offset, the same parameters as `bind_buffer_memory`, but batched.

Use `BindBufferMemoryDeviceGroupInfo` in the pNext chain to bind
memory for specific devices in a device group (multi-GPU). For
single-GPU usage, `bind_buffer_memory` and `bind_buffer_memory2`
are equivalent.
