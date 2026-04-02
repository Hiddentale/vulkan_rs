# Usage Notes

Clears one or more regions of a depth/stencil image to a specified
depth and stencil value. The image must be in
`TRANSFER_DST_OPTIMAL` or `GENERAL` layout.

For most rendering, clearing via `load_op = CLEAR` on the
depth/stencil attachment is preferred, it lets tile-based GPUs
avoid a separate clear pass. Use this command only when you need to
clear a depth/stencil image outside a render pass.

The `image_subresource_range` must reference the appropriate aspect
flags (`DEPTH`, `STENCIL`, or both).

Must be recorded outside a render pass.
