# Usage Notes

Assigns a name to a Vulkan object for debugging. This is the
legacy `VK_EXT_debug_marker` equivalent of
`set_debug_utils_object_name_ext`.

`DebugMarkerObjectNameInfoEXT` uses the old
`DebugReportObjectTypeEXT` enum to identify object types.

Superseded by `VK_EXT_debug_utils`. Prefer
`set_debug_utils_object_name_ext` for new code.
