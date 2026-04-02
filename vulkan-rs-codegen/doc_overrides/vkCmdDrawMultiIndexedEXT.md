# Usage Notes

Draws multiple indexed draw calls from an array of
`MultiDrawIndexedInfoEXT` (first_index, index_count,
vertex_offset triples). More efficient than separate
`cmd_draw_indexed` calls.

An optional `p_vertex_offset` overrides all per-draw vertex
offsets with a single value.

Requires `VK_EXT_multi_draw` and the `multiDraw` feature.
