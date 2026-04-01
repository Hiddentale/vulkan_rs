# Usage Notes

Dynamically enables or disables stencil testing. Only takes effect
if the pipeline was created with
`DYNAMIC_STATE_STENCIL_TEST_ENABLE`.

When disabled, fragments pass the stencil test unconditionally and
no stencil writes occur.

Core dynamic state in Vulkan 1.3.
