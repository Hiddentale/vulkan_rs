# Usage Notes

Dynamically enables or disables depth testing. Only takes effect if
the pipeline was created with `DYNAMIC_STATE_DEPTH_TEST_ENABLE`.

When disabled, all fragments pass the depth test regardless of the
depth buffer contents. Useful for UI overlays, skyboxes, or
full-screen post-processing passes.

Core dynamic state in Vulkan 1.3.
