# Usage Notes

Queries whether a queue family on the physical device supports
presentation to the Win32 desktop compositor. Windows only.
Check this before creating a swapchain to ensure the queue can
present.

Requires `VK_KHR_win32_surface`.
