# Usage Notes

Dynamically remaps input attachment indices within a dynamic
rendering instance. Allows fragment shaders to read from different
colour or depth/stencil attachments without changing the pipeline.

Paired with `cmd_set_rendering_attachment_locations` to enable
flexible attachment routing in multi-pass rendering with dynamic
rendering.

Requires `dynamic_rendering_local_read` feature. Core in
Vulkan 1.4.
