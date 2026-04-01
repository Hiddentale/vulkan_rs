# Usage Notes

Enumerates displays attached to a physical device. Each returned
`DisplayPropertiesKHR` contains the display handle, name,
physical dimensions, resolution, supported transforms, and
whether the display supports per-plane reordering.

This is the entry point for the `VK_KHR_display` extension,
which provides direct display output without a window system
(useful for embedded, VR, kiosk, and fullscreen applications).

After enumerating displays, query their modes with
`get_display_mode_properties_khr` and planes with
`get_physical_device_display_plane_properties_khr`.

Prefer `get_physical_device_display_properties2_khr` when
`VK_KHR_get_display_properties2` is available, it supports
extensible output via `pNext`.
