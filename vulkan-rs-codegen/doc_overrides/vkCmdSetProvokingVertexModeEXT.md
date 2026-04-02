# Usage Notes

Dynamically sets which vertex in a primitive is the provoking
vertex (the vertex whose flat-shaded attributes are used):
- `FIRST_VERTEX`: first vertex of the primitive (Vulkan default).
- `LAST_VERTEX`: last vertex (OpenGL convention).

Requires `VK_EXT_provoking_vertex` and the
`provokingVertexLast` feature for `LAST_VERTEX` mode.

Provided by `VK_EXT_extended_dynamic_state3`.
