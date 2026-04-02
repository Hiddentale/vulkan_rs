# Usage Notes

Creates a swapchain, a queue of presentable images tied to a
surface (window). The swapchain is the bridge between rendering and
display.

**Key parameters**:

- **`min_image_count`**: request at least this many images. For
  double buffering use 2, for triple buffering use 3. Query
  `get_physical_device_surface_capabilities_khr` for the supported
  range.
- **`image_format` / `image_color_space`**: pick a pair from
  `get_physical_device_surface_formats_khr`. `B8G8R8A8_SRGB` +
  `SRGB_NONLINEAR` is the most portable.
- **`present_mode`**: `FIFO` (vsync, always supported), `MAILBOX`
  (low-latency triple buffering), `IMMEDIATE` (no vsync, tearing).
- **`pre_transform`**: set to `current_transform` from surface
  capabilities to avoid an extra composition blit.
- **`old_swapchain`**: when recreating after a resize, pass the old
  swapchain here. The driver can reuse internal resources.

**Swapchain recreation** is required when the surface size changes
(window resize) or when `acquire_next_image_khr` returns
`VK_ERROR_OUT_OF_DATE_KHR`. Destroy the old swapchain after
creating the new one.

# Guide

See [Hello Triangle, Part 2](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-2.html) in the vulkan_rs guide.
