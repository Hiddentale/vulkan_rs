# Usage Notes

Binds an index buffer for subsequent indexed draw calls
(`cmd_draw_indexed`, `cmd_draw_indexed_indirect`).

**Index type**:

- `INDEX_TYPE_UINT16`: 2 bytes per index. Good for meshes with
  fewer than 65536 vertices, saves memory bandwidth.
- `INDEX_TYPE_UINT32`: 4 bytes per index. Required for large meshes.
- `INDEX_TYPE_UINT8_KHR` (extension): 1 byte per index for very
  small meshes.

The buffer must have been created with `BUFFER_USAGE_INDEX_BUFFER`.
The `offset` must be a multiple of the index type size (2 for
UINT16, 4 for UINT32).

Only one index buffer can be bound at a time, binding a new one
replaces the previous binding.
