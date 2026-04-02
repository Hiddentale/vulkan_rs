# Usage Notes

Sets the offsets into bound descriptor buffers for one or more
descriptor set slots. Each pair of (buffer_index, offset) maps
a descriptor set to a region of a previously bound descriptor
buffer.

Must be called after `cmd_bind_descriptor_buffers_ext`.

For the pNext-extensible variant, see
`cmd_set_descriptor_buffer_offsets2_ext`.

Requires `VK_EXT_descriptor_buffer`.
