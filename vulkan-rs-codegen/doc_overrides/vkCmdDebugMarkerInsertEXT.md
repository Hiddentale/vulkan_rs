# Usage Notes

Inserts a single-point debug marker into the command buffer.
This is the legacy `VK_EXT_debug_marker` equivalent of
`cmd_insert_debug_utils_label_ext`.

`DebugMarkerMarkerInfoEXT` specifies a name and optional RGBA
color.

Superseded by `VK_EXT_debug_utils`. Prefer
`cmd_insert_debug_utils_label_ext` for new code.
