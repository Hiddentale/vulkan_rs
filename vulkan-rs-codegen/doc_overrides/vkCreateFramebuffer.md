# Usage Notes

A framebuffer binds concrete image views to the attachment slots
defined by a render pass. The number and format of attachments must
match the render pass exactly, mismatches cause validation errors.

**Dimensions**: `width`, `height`, and `layers` must be less than
or equal to the corresponding dimensions of every attached image
view. They define the renderable area for `cmd_begin_render_pass`.

**Lifetime**: the framebuffer must stay alive for the entire
duration of any render pass instance that uses it. In practice,
framebuffers are typically recreated when the swapchain is resized.

**Imageless framebuffers** (Vulkan 1.2+): create the framebuffer
with `FRAMEBUFFER_CREATE_IMAGELESS` and no attachments. Concrete
image views are then supplied at `cmd_begin_render_pass` time via
`RenderPassAttachmentBeginInfo`. This avoids recreating framebuffers
on swapchain resize.

# Guide

See [Render Passes & Framebuffers](https://hiddentale.github.io/vulkan_rs/concepts/render-passes.html) in the vulkan_rs guide.
