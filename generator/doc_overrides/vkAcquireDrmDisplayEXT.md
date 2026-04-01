# Usage Notes

Acquires exclusive control of a DRM display for direct rendering.
Takes a DRM file descriptor and a display handle. Release with
`release_display_ext` when finished.

Requires `VK_EXT_acquire_drm_display`. Linux only.
