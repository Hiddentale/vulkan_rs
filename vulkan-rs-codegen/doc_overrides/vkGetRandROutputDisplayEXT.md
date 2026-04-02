# Usage Notes

Maps an X11 RandR output to a Vulkan `DisplayKHR` handle. Use
this to identify which Vulkan display corresponds to a specific
RandR output when doing direct display rendering.

Requires `VK_EXT_acquire_xlib_display`. Linux/X11 only.
