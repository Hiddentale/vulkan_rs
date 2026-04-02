# Usage Notes

Opens a debug label region on a queue. All submissions between
this call and the matching `queue_end_debug_utils_label_ext` are
grouped under the label in GPU debuggers.

Unlike `cmd_begin_debug_utils_label_ext` (which operates inside
a command buffer), this groups entire queue submissions.

Requires `VK_EXT_debug_utils`.
