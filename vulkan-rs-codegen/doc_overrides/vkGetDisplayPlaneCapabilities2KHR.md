# Usage Notes

Extensible version of `get_display_plane_capabilities_khr`.
Takes `DisplayPlaneInfo2KHR` (with `pNext` for input extensions)
and writes to `DisplayPlaneCapabilities2KHR` (with `pNext` for
output extensions).

Provided by `VK_KHR_get_display_properties2`. Prefer this over
the v1 query when available.
