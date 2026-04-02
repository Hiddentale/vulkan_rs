# Usage Notes

Dynamically enables or disables line stippling. When enabled,
lines are drawn with a repeating pattern defined by
`cmd_set_line_stipple` (core 1.4) or `cmd_set_line_stipple_ext`.

Requires `VK_EXT_line_rasterization` and the `stippledLineEnable`
feature for the active line rasterization mode.

Provided by `VK_EXT_extended_dynamic_state3`.
