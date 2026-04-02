# Usage Notes

Resolves (downsamples) a multisample image into a single-sample
image. Typically used to produce the final single-sample result from
a multisampled colour attachment.

Both images must be in appropriate transfer layouts
(`TRANSFER_SRC_OPTIMAL` and `TRANSFER_DST_OPTIMAL` respectively).
The source must be multisampled; the destination must be
single-sample. Formats must be identical.

For resolving inside a render pass, use `resolve_attachment` in the
subpass description instead, it is more efficient on tile-based
GPUs because the resolve happens in-tile.

Must be recorded outside a render pass. For Vulkan 1.3+, prefer
`cmd_resolve_image2`.
