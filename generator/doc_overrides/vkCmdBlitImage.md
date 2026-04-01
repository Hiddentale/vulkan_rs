# Usage Notes

Copies a region between two images with optional scaling and format
conversion. Unlike `cmd_copy_image`, blit supports different source
and destination extents (scaling) and applies a filter.

**Filters**:

- `FILTER_NEAREST`: no interpolation. Fast, but produces blocky
  results when scaling.
- `FILTER_LINEAR`: bilinear interpolation. Smooth scaling. Requires
  the format to support `FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR`.

Common uses:

- **Mipmap generation**: blit each mip level from the previous one,
  halving dimensions each step.
- **Resolve or downscale**: blit a high-resolution offscreen image
  to a smaller swapchain image.

Both images must be in appropriate transfer layouts. Must be recorded
outside a render pass. For Vulkan 1.3+, prefer `cmd_blit_image2`.

Not supported for depth/stencil or compressed formats. Use
`cmd_copy_image` or `cmd_resolve_image` for those.
