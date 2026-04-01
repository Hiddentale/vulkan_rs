# Usage Notes

Queries whether a specific combination of image format, type, tiling,
usage, and flags is supported on this device, and if so, what limits
apply.

Use this to validate image creation parameters before calling
`create_image`. For example, check whether a format supports the
`COLOR_ATTACHMENT` usage with optimal tiling at your desired
resolution.

The returned `ImageFormatProperties` includes:

- **`max_extent`**: maximum dimensions for this combination.
- **`max_mip_levels`**: maximum mipmap levels.
- **`max_array_layers`**: maximum array layers.
- **`sample_counts`**: supported multisample counts.
- **`max_resource_size`**: maximum total bytes.

Returns `VK_ERROR_FORMAT_NOT_SUPPORTED` if the combination is not
supported at all, this is not a fatal error, just a "no".

For extended queries (Vulkan 1.1+), use
`get_physical_device_image_format_properties2` which supports
chaining external memory and YCBCR properties.
