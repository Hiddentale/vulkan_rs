# Usage Notes

Queries the sparse image format properties for a specific format,
image type, sample count, usage, and tiling combination. Returns
information about the sparse texel block dimensions and flags.

Only relevant if you intend to use sparse images
(`IMAGE_CREATE_SPARSE_*`). For non-sparse images, this is not
needed.

If the combination does not support sparse residency, an empty list
is returned. Check `physical_device_features.sparse_residency_*`
features before attempting sparse image creation.

For extended queries (Vulkan 1.1+), use
`get_physical_device_sparse_image_format_properties2`.
