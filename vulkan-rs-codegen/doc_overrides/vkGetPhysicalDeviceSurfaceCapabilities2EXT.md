# Usage Notes

Queries surface capabilities with additional output fields
compared to `get_physical_device_surface_capabilities_khr`.
Returns a `SurfaceCapabilities2EXT` that includes shared
present mode support flags.

Prefer `get_physical_device_surface_capabilities2_khr` (KHR)
for general use; this EXT variant is primarily for
`VK_EXT_display_surface_counter` integration.

Requires `VK_EXT_display_surface_counter`.
