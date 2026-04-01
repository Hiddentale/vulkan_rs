# Usage Notes

Indirect-count variant of `cmd_draw_mesh_tasks_nv`. Reads both
the draw parameters and the draw count from GPU buffers, enabling
fully GPU-driven mesh shader dispatch where the number of draws
is determined at runtime.

Requires `VK_NV_mesh_shader`.
