# Usage Notes

Records an indexed draw call, the standard way to draw meshes.
Indices are read from the bound index buffer, and each index is used
to fetch vertex attributes from the bound vertex buffers.

**Parameters**:

- `index_count`: number of indices to read.
- `instance_count`: number of instances. Use 1 for non-instanced.
- `first_index`: offset into the index buffer (in units of indices,
  not bytes).
- `vertex_offset`: added to each index value before fetching the
  vertex. Useful for packing multiple meshes into a single vertex
  buffer at different offsets.
- `first_instance`: offset into instance data.

Indexed drawing reuses vertices via the index buffer, saving memory
and bandwidth compared to non-indexed draws for any mesh with shared
vertices (which is nearly all of them).
