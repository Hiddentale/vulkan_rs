# Usage Notes

Queries the memory layout (offset, size, row pitch, array pitch,
depth pitch) of a specific subresource within a linear-tiling
image. Only valid for images created with `IMAGE_TILING_LINEAR`.
For optimal-tiling images, use `get_image_subresource_layout2`.
