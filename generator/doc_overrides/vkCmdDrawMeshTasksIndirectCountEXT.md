# Usage Notes

Count-indirect version of mesh shader dispatch. Both the dispatch
parameters and the draw count are read from GPU buffers, enabling
fully GPU-driven mesh shader workloads.

`max_draw_count` caps the count read from the count buffer.

Requires `VK_EXT_mesh_shader`.
