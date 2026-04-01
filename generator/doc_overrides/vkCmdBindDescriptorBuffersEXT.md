# Usage Notes

Binds one or more descriptor buffers to a command buffer. Each
`DescriptorBufferBindingInfoEXT` specifies a buffer address and
usage (resource descriptors, sampler descriptors, or push
descriptors).

After binding, use `cmd_set_descriptor_buffer_offsets_ext` to
point specific descriptor sets at offsets within the bound buffers.

Descriptor buffers are an alternative to descriptor sets/pools
that stores descriptors inline in buffer memory.

Requires `VK_EXT_descriptor_buffer`.
