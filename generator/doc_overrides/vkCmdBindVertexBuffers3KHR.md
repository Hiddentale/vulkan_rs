# Usage Notes

Device-address variant of `cmd_bind_vertex_buffers`. Binds vertex
buffers for subsequent draw commands using device addresses instead
of buffer handles.

Each `BindVertexBuffer3InfoKHR` specifies a device address, size,
and stride for one binding slot starting at `first_binding`.

Supersedes `cmd_bind_vertex_buffers`, `cmd_bind_vertex_buffers2`,
and the core 1.4 equivalent when using
`VK_KHR_device_address_commands`.
