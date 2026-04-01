# Usage Notes

Opens a debug marker region in a command buffer. This is the
legacy `VK_EXT_debug_marker` equivalent of
`cmd_begin_debug_utils_label_ext`.

`DebugMarkerMarkerInfoEXT` specifies a name and optional RGBA
color. Close with `cmd_debug_marker_end_ext`.

Superseded by `VK_EXT_debug_utils`. Prefer
`cmd_begin_debug_utils_label_ext` for new code.
