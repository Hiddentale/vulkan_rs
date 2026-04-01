# Usage Notes

Dynamically enables or disables color blending for each color
attachment. Pass a slice of `Bool32` values, one per attachment
starting at `first_attachment`.

When blending is disabled for an attachment, the fragment color
is written directly (no src/dst blending).

Provided by `VK_EXT_extended_dynamic_state3`.
