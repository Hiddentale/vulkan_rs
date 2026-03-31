# Usage Notes

Frees a device memory allocation. All buffers and images bound to
this memory must already be destroyed — freeing memory while objects
are still bound is undefined behaviour.

If the memory is currently mapped, it is implicitly unmapped before
being freed. You do not need to call `unmap_memory` first, although
doing so explicitly is a common defensive practice.

Vulkan has a per-device allocation limit
(`max_memory_allocation_count`, often 4096). Sub-allocating from
large blocks and freeing them as a group keeps you well within this
limit.
