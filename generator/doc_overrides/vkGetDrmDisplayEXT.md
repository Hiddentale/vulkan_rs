# Usage Notes

Maps a DRM connector to a Vulkan `DisplayKHR` handle. Takes a
DRM file descriptor and connector ID, and returns the
corresponding display. Use this to identify which Vulkan display
corresponds to a specific DRM output.

Requires `VK_EXT_acquire_drm_display`. Linux only.
