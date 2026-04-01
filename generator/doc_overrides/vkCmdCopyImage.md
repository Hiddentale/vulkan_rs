# Usage Notes

Copies texel data between two images. Both images must have been
created with `TRANSFER_SRC` and `TRANSFER_DST` usage respectively,
and must be in compatible layouts (`TRANSFER_SRC_OPTIMAL` /
`TRANSFER_DST_OPTIMAL` or `GENERAL`).

The source and destination formats must be identical, or both must
be in the same size-compatibility class. For format conversion, use
`cmd_blit_image` instead.

Copy operates on raw texel blocks, no filtering or scaling. The
extent must be aligned to the texel block size for compressed
formats.

Must be recorded outside a render pass. For Vulkan 1.3+, prefer
`cmd_copy_image2`.
