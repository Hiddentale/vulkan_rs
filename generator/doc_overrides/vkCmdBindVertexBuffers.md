# Usage Notes

Binds one or more vertex buffers to input binding slots for
subsequent draw calls.

**`first_binding`**: the binding slot index to start at. Binding
slots are defined in the pipeline's vertex input state. Multiple
buffers can be bound to consecutive slots in a single call.

**Interleaved vs separate**: a single buffer with interleaved
attributes (position + normal + UV) uses one binding slot. Separate
attribute streams (one buffer per attribute) use multiple slots.
Interleaved is generally more cache-friendly.

Buffers must have been created with `BUFFER_USAGE_VERTEX_BUFFER`.
The `offsets` array specifies the byte offset within each buffer
where vertex data starts.

For Vulkan 1.3+, `cmd_bind_vertex_buffers2` also lets you set
buffer sizes and strides dynamically.
