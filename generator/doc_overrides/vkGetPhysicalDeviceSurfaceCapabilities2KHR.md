# Usage Notes

Extensible version of
`get_physical_device_surface_capabilities_khr`. Takes a
`PhysicalDeviceSurfaceInfo2KHR` input and writes to
`SurfaceCapabilities2KHR`, both supporting `pNext` chains.

Chain `SurfaceProtectedCapabilitiesKHR` or other extension
structs into the output `pNext` to query additional capabilities
not available through the v1 query.

Provided by `VK_KHR_get_surface_capabilities2`. Prefer this over
the v1 query when available.
