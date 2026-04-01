# Usage Notes

Returns the total byte size required to store all descriptors for
the given descriptor set layout in a descriptor buffer.

Use this to allocate the correct amount of buffer memory for each
descriptor set, then write individual descriptors at offsets
obtained from `get_descriptor_set_layout_binding_offset_ext`.

Requires `VK_EXT_descriptor_buffer`.
