# Usage Notes

Returns the memory requirements (size, alignment, compatible memory
type bits) for a buffer. Must be called before `bind_buffer_memory`.

The returned `memory_type_bits` is a bitmask where bit *i* is set if
memory type *i* (from `get_physical_device_memory_properties`) is
compatible. Pick a type that satisfies both this mask and your
desired properties (`HOST_VISIBLE`, `DEVICE_LOCAL`, etc.).

The `alignment` value must be respected when sub-allocating: the
offset passed to `bind_buffer_memory` must be a multiple of it.

For Vulkan 1.1+, prefer `get_buffer_memory_requirements2` which
supports dedicated allocation queries via
`MemoryDedicatedRequirements`.
