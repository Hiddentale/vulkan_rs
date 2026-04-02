# Usage Notes

Vulkan 1.2 version of `create_render_pass` that uses extensible
`RenderPassCreateInfo2`, `AttachmentDescription2`,
`SubpassDescription2`, and `SubpassDependency2` structs.

Key additions over the 1.0 version:

- **View masks**: `SubpassDescription2::view_mask` enables multiview
  rendering where a single draw call renders to multiple layers
  simultaneously (e.g. VR left/right eyes).
- **Fragment density map**: chain
  `RenderPassFragmentDensityMapCreateInfoEXT` for variable-rate
  shading via density maps.
- **Depth/stencil resolve**: `SubpassDescriptionDepthStencilResolve`
  enables automatic depth/stencil resolve at the end of a subpass.

Prefer this over `create_render_pass` when targeting Vulkan 1.2+.
For Vulkan 1.3+, consider dynamic rendering (`cmd_begin_rendering`)
which avoids render pass objects entirely.
