# Usage Notes

Dynamically enables or disables rasterizer discard. Only takes
effect if the pipeline was created with
`DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE`.

When enabled, primitives are discarded before rasterisation, no
fragments are generated and no colour/depth output is produced. The
vertex and geometry shader stages still execute.

Use cases:

- **Transform feedback only**: capture transformed vertices without
  rendering.
- **Occlusion pre-pass**: skip fragment shading when only the depth
  or stencil output matters (though depth writes still require
  rasterisation).

Core dynamic state in Vulkan 1.3.
