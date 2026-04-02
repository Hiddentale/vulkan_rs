# Usage Notes

Vulkan 1.1 version of `get_buffer_memory_requirements` that supports
extensible output structs via pNext.

Chain `MemoryDedicatedRequirements` to query whether the driver
prefers or requires a dedicated allocation for this buffer. If
`prefers_dedicated_allocation` is true, allocating a dedicated
`DeviceMemory` for this buffer may improve performance.

The base `MemoryRequirements` (size, alignment, memory type bits) is
identical to what `get_buffer_memory_requirements` returns.
