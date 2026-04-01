# Usage Notes

Vulkan 1.3 command that queries memory requirements for a buffer
**without creating it first**. Pass a `DeviceBufferMemoryRequirements`
containing the hypothetical `BufferCreateInfo`.

This lets you pre-plan memory allocations before creating any
objects, useful for memory allocation strategies that need to know
sizes and alignments up front.

The returned requirements are identical to what
`get_buffer_memory_requirements2` would return for an actual buffer
created with the same parameters.
