# Usage Notes

Vulkan 1.2 version of `cmd_begin_render_pass` that takes an
additional `SubpassBeginInfo` parameter specifying the subpass
contents mode.

The extensible structs allow chaining `RenderPassAttachmentBeginInfo`
for imageless framebuffers, concrete image views are supplied at
begin time rather than at framebuffer creation time.

Prefer this over `cmd_begin_render_pass` when targeting Vulkan 1.2+.
For Vulkan 1.3+, consider `cmd_begin_rendering` (dynamic rendering)
which eliminates render pass and framebuffer objects entirely.
