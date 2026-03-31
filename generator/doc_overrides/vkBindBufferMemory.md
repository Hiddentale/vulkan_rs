# Usage Notes

Binds a `DeviceMemory` allocation (or a region of one) to a buffer.
Must be called before the buffer is used in any command.

**Requirements**:

1. Call `get_buffer_memory_requirements` first.
2. The `memory_type_index` used for allocation must have a bit set
   in the returned `memory_type_bits` mask.
3. `memory_offset` must be a multiple of the returned `alignment`.
4. `memory_offset + size` must not exceed the allocation size.

**Sub-allocation**: multiple buffers can share one `DeviceMemory`
allocation at different offsets. This is strongly recommended —
drivers have a per-allocation limit (`max_memory_allocation_count`,
often 4096) and each allocation has overhead.

Once bound, the memory binding cannot be changed for the lifetime of
the buffer. Destroy the buffer before freeing its backing memory.
