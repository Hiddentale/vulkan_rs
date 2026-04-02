# Usage Notes

Extensible version of
`get_physical_device_surface_formats_khr`. Returns
`SurfaceFormat2KHR` with `pNext` support, allowing extensions
to attach additional per-format information.

Takes `PhysicalDeviceSurfaceInfo2KHR` as input so you can query
formats for a specific surface configuration (e.g., with
full-screen exclusive info chained in).

Provided by `VK_KHR_get_surface_capabilities2`. Prefer this over
the v1 query when available.
