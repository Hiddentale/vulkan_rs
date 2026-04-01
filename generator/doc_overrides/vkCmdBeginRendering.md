# Usage Notes

Begins dynamic rendering, a Vulkan 1.3 alternative to render pass
objects that specifies attachments inline at command recording time.

**Advantages over render passes**:

- No `RenderPass` or `Framebuffer` objects to create and manage.
- Attachments are specified directly as image views in
  `RenderingInfo`.
- Simpler code for applications that do not benefit from tile-based
  subpass optimisations.

**`RenderingInfo`** specifies:

- **Colour attachments**: image views, load/store ops, clear values.
- **Depth attachment**: optional, with its own load/store ops.
- **Stencil attachment**: optional, can share the same image view as
  depth.
- **Render area and layer count**.

**Flags**:

- `RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS`: draw commands come
  from secondary command buffers.
- `RENDERING_SUSPENDING` / `RENDERING_RESUMING`: split rendering
  across multiple command buffers.

Graphics pipelines used with dynamic rendering must be created with
`PipelineRenderingCreateInfo` instead of a render pass handle.
