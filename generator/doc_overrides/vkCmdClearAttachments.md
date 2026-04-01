# Usage Notes

Clears one or more attachment regions **inside** an active render
pass. Unlike `load_op = CLEAR` (which clears the entire attachment
at render pass begin), this clears arbitrary rectangular regions
mid-render-pass.

Use cases:

- Clear a sub-region of a colour attachment (e.g. a UI panel
  background).
- Clear the stencil buffer for a specific screen region.

Each `ClearAttachment` specifies which attachment to clear (colour
index, depth, or stencil) and the clear value. Each `ClearRect`
defines the pixel rectangle and layer range.

For whole-attachment clears, prefer `load_op = CLEAR`, it is
always at least as fast and often faster on tile-based hardware.
