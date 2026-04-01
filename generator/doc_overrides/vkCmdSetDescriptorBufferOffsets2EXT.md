# Usage Notes

Extended version of `cmd_set_descriptor_buffer_offsets_ext` that
takes a `SetDescriptorBufferOffsetsInfoEXT` struct with pNext
extensibility.

Sets the offsets into bound descriptor buffers for the specified
pipeline layout. Each offset points to the start of a descriptor
set's data within the bound descriptor buffer.

Provided by `VK_KHR_maintenance6` (for the pNext-extensible
variant of the `VK_EXT_descriptor_buffer` command).
