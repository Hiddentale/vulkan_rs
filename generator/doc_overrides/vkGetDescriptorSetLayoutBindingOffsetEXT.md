# Usage Notes

Returns the byte offset of a specific binding within the
descriptor buffer layout for the given descriptor set layout.

Use this to compute where to write a descriptor with
`get_descriptor_ext` within the buffer region for a set.

Requires `VK_EXT_descriptor_buffer`.
