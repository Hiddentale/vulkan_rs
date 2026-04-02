# Usage Notes

Clears one or more regions of a colour image to a specified value.
The image must be in `TRANSFER_DST_OPTIMAL` or `GENERAL` layout.

This is an explicit clear outside a render pass. For clears inside
a render pass, use `load_op = CLEAR` in the attachment description
or `cmd_clear_attachments`, both are typically faster because the
driver can integrate them with tile-based rendering.

The clear value is a `ClearColorValue` union: either four `float32`,
`int32`, or `uint32` values depending on the image format.

Must be recorded outside a render pass.
