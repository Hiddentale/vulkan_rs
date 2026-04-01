# Usage Notes

Pushes inline data to the command buffer for use in shaders. The
`PushDataInfoEXT` specifies the pipeline layout, stage flags,
offset, and data bytes.

Similar to push constants but used with the descriptor heap
model. Data is accessible in shaders via the push data mechanism.

Provided by `VK_EXT_descriptor_heap`.
