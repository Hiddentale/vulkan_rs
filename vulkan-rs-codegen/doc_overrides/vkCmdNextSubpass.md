# Usage Notes

Advances to the next subpass within a render pass. Subpass
transitions allow the driver to resolve dependencies between
subpasses, for example, reading a colour attachment written in
the previous subpass as an input attachment.

The `contents` parameter has the same meaning as in
`cmd_begin_render_pass`: `INLINE` or `SECONDARY_COMMAND_BUFFERS`.

Multi-subpass render passes are an optimisation for tile-based GPUs
where they can keep data on-chip between subpasses. On desktop GPUs
the benefit is smaller. Many applications use a single subpass and
handle inter-pass dependencies with explicit pipeline barriers.

For Vulkan 1.2+, prefer `cmd_next_subpass2`.
