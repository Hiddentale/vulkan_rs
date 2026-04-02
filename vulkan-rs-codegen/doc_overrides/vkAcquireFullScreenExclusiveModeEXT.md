# Usage Notes

Acquires full-screen exclusive mode for a swapchain, giving the
application direct control over the display output. This can
reduce latency and enable adaptive sync.

The swapchain must have been created with
`SurfaceFullScreenExclusiveInfoEXT` in its pNext chain.
Release with `release_full_screen_exclusive_mode_ext`.

Requires `VK_EXT_full_screen_exclusive`. Windows only.
