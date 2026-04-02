# Usage Notes

Records a non-indexed draw call. Vertices are generated sequentially
from `first_vertex` to `first_vertex + vertex_count - 1`.

**Parameters**:

- `vertex_count`: number of vertices to draw.
- `instance_count`: number of instances. Use 1 for non-instanced
  drawing.
- `first_vertex`: offset into the vertex buffer (added to
  `gl_VertexIndex` in the shader).
- `first_instance`: offset into instance data (added to
  `gl_InstanceIndex`). Requires the `first_instance` feature if
  non-zero.

For indexed geometry (the common case for meshes), use
`cmd_draw_indexed` instead. `cmd_draw` is typically used for
full-screen quads, procedural geometry, or particle systems where
vertices are generated in the shader.

# Guide

See [Command Buffers](https://hiddentale.github.io/vulkan_rs/concepts/command-buffers.html) in the vulkan_rs guide.
