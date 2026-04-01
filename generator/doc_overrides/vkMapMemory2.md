# Usage Notes

Vulkan 1.4 version of `map_memory` that uses an extensible
`MemoryMapInfo` struct. Supports additional flags via
`MemoryMapFlags` that are not available in the original 1.0 version.

`MEMORY_MAP_PLACED` (if supported) allows mapping at a
caller-specified virtual address, useful for deterministic replay
tools.

For most applications, `map_memory` and `map_memory2` are
equivalent. Prefer this when targeting Vulkan 1.4+.
