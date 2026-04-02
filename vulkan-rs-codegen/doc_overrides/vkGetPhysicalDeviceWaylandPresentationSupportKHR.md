# Usage Notes

Queries whether a queue family supports presentation to a
Wayland compositor. Linux/Wayland only. Check this before
creating a swapchain to ensure the queue can present.

Requires `VK_KHR_wayland_surface`.
