# Usage Notes

Begins a render pass instance. All subsequent drawing commands are
recorded within this render pass until `cmd_end_render_pass`.

**`render_pass_begin_info`** specifies:

- **Render pass and framebuffer**: which render pass to use and
  which concrete image views are bound.
- **Render area**: the pixel region to render. Should match the
  framebuffer extent for best performance. Misalignment with the
  render area granularity can cause overhead on tile-based GPUs.
- **Clear values**: one per attachment with `load_op = CLEAR`. The
  array must include entries for all attachments (use a dummy value
  for non-cleared attachments).

**`contents`**:

- `SUBPASS_CONTENTS_INLINE`: draw commands are recorded directly
  in this command buffer.
- `SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS`: draw commands come
  from secondary command buffers via `cmd_execute_commands`.

For Vulkan 1.2+, `cmd_begin_render_pass2` accepts a `SubpassBeginInfo`.
For Vulkan 1.3+, consider dynamic rendering (`cmd_begin_rendering`)
which avoids render pass and framebuffer objects entirely.

# Guide

See [Render Passes & Framebuffers](https://hiddentale.github.io/vulkan_rs/concepts/render-passes.html) in the vulkan_rs guide.
