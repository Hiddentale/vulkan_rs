# Usage Notes

Dynamically enables or disables attachment feedback loops for
specific image aspects (color, depth, stencil). When enabled,
shaders can both read from and write to the same attachment
within a render pass.

Use with care, feedback loops create read-after-write hazards.
The implementation handles coherency when this flag is set.

Requires `VK_EXT_attachment_feedback_loop_dynamic_state`.
