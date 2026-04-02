# Usage Notes

Legacy NV path for querying external image format properties.
Takes the full set of image creation parameters plus an external
handle type and returns compatibility information. Prefer the
core `get_physical_device_image_format_properties2` with
`PhysicalDeviceExternalImageFormatInfo` in the pNext chain.

Requires `VK_NV_external_memory_capabilities`.
