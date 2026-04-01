# Usage Notes

Queries which Vulkan memory types are compatible with importing
a Metal resource handle as external memory. Use before allocating
device memory to determine valid memory type bits.

Requires `VK_EXT_external_memory_metal`. macOS/iOS only.
