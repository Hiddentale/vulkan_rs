# Usage Notes

Vulkan 1.4 version of `unmap_memory` that uses an extensible
`MemoryUnmapInfo` struct. Supports `MEMORY_UNMAP_RESERVE` (if
available) to keep the virtual address range reserved after
unmapping, useful for placed mappings.

For most applications, `unmap_memory` and `unmap_memory2` are
equivalent. Prefer this when targeting Vulkan 1.4+.
