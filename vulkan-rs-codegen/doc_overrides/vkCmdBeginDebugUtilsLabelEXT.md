# Usage Notes

Opens a debug label region in a command buffer. All commands
recorded between this call and the matching
`cmd_end_debug_utils_label_ext` are grouped under the label in
GPU debuggers (RenderDoc, Nsight).

`DebugUtilsLabelEXT` specifies a name and optional RGBA color
for the region.

Labels can nest. Every begin must have a matching end within the
same command buffer.

Requires `VK_EXT_debug_utils`.
