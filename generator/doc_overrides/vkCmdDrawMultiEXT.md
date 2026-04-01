# Usage Notes

Draws multiple non-indexed draw calls from an array of
`MultiDrawInfoEXT` (first_vertex, vertex_count pairs). More
efficient than issuing separate `cmd_draw` calls because the
driver can batch them.

Requires `VK_EXT_multi_draw` and the `multiDraw` feature.
