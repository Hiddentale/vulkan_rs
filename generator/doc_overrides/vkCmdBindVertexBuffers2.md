# Usage Notes

Vulkan 1.3 version of `cmd_bind_vertex_buffers` that additionally
accepts optional buffer sizes and strides.

**Sizes**: if provided, the driver knows where each buffer ends and
can perform bounds checking. Pass null to use the full buffer size.

**Strides**: if provided, overrides the stride specified in the
pipeline's vertex input state. This enables dynamic vertex stride
without creating separate pipeline permutations. Pass null to use
the pipeline's static stride.

Prefer this over `cmd_bind_vertex_buffers` when targeting
Vulkan 1.3+.
