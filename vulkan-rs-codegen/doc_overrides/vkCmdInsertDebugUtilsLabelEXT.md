# Usage Notes

Inserts a single-point debug label into the command buffer
(as opposed to a begin/end region). Useful for marking specific
events like "shadow pass complete" or "post-process start" that
don't span a range of commands.

Appears as a marker in GPU debuggers (RenderDoc, Nsight).

Requires `VK_EXT_debug_utils`.
