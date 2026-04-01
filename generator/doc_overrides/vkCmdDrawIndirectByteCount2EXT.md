# Usage Notes

Device-address variant of `cmd_draw_indirect_byte_count_ext`.
Draws vertices using a byte count from a transform feedback
counter, with the counter buffer specified via device address
instead of a buffer handle.

`counter_offset` is the byte offset within the counter value
to account for any header. `vertex_stride` determines how many
bytes each vertex consumes.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_transform_feedback`.
