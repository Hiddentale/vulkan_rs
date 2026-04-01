# Usage Notes

Indirect version of `cmd_draw_mesh_tasks_ext`. Reads mesh shader
dispatch parameters from a buffer. Each indirect command contains
group counts (x, y, z).

`draw_count` specifies how many indirect commands to execute.
`stride` is the byte stride between commands in the buffer.

Requires `VK_EXT_mesh_shader`.
