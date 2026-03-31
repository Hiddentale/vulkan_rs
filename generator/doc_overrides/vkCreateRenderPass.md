# Usage Notes

A render pass describes the attachments, subpasses, and dependencies
used during rendering. It does not reference actual images — those
are bound later via a framebuffer.

Key design points:

- **`load_op` / `store_op`**: use `DONT_CARE` for attachments whose
  prior contents are irrelevant (e.g. a transient depth buffer). This
  lets tile-based GPUs skip loads/stores, which is significant on
  mobile.
- **`initial_layout` / `final_layout`**: Vulkan inserts implicit layout
  transitions at render pass boundaries. Set these to match your actual
  usage to avoid unnecessary transitions. `UNDEFINED` for `initial_layout`
  is fine when `load_op` is `CLEAR` or `DONT_CARE`.
- **Subpass dependencies**: the implicit external dependencies
  (`VK_SUBPASS_EXTERNAL`) are often insufficient. Add explicit
  dependencies when subsequent passes read the output.

For dynamic rendering (Vulkan 1.3+), consider `cmd_begin_rendering`
instead, which avoids the need for render pass and framebuffer objects.
