# Usage Notes

Device-address variant of `cmd_draw_indexed_indirect_count`. Reads
indexed draw parameters and the draw count from device addresses
instead of buffer handles.

The `DrawIndirectCount2InfoKHR` specifies both the draw parameter
address and the count address, along with `max_draw_count` and
stride.

Requires `VK_KHR_device_address_commands`.
