# Usage Notes

Ends the current render pass instance. After this call, the
implicit layout transitions specified by each attachment's
`final_layout` are applied.

No draw commands may be recorded after this until a new render pass
is begun (or dynamic rendering is started).

For Vulkan 1.2+, prefer `cmd_end_render_pass2`.
