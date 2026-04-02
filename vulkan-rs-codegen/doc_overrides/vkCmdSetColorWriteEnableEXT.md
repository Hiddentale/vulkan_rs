# Usage Notes

Dynamically enables or disables color writing for each color
attachment. Pass a slice of `Bool32` values, one per attachment.

When color write is disabled for an attachment, no color output
is written regardless of blend state.

Unlike `cmd_set_color_write_mask_ext` (which controls per-channel
masking), this is a simple on/off toggle per attachment.

Provided by `VK_EXT_color_write_enable`.
