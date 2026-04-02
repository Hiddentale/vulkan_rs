# Usage Notes

Queries which memory types are compatible with importing a host
pointer as external memory. Use this before allocating device
memory backed by a host-allocated buffer to determine valid
memory type bits.

Requires `VK_EXT_external_memory_host`.
