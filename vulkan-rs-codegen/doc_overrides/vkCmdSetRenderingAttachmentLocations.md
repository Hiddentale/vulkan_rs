# Usage Notes

Dynamically remaps colour attachment indices within a dynamic
rendering instance. Allows fragment shader outputs to target
different attachment slots without changing the pipeline.

This is useful when the same shader outputs different data to
different attachments depending on the rendering pass (e.g.
G-buffer vs forward).

Requires `dynamic_rendering_local_read` feature. Core in
Vulkan 1.4.
