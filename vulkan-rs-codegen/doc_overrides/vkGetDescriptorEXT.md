# Usage Notes

Writes a descriptor directly into caller-provided memory.
`DescriptorGetInfoEXT` specifies the descriptor type and resource
(buffer, image, sampler, etc.). The descriptor is written to
`p_descriptor` and must be `data_size` bytes.

Query the required size per descriptor type with
`PhysicalDeviceDescriptorBufferPropertiesEXT`.

This is the core operation of descriptor buffers, instead of
allocating descriptor sets, you write descriptors directly into
mapped buffer memory.

Requires `VK_EXT_descriptor_buffer`.
