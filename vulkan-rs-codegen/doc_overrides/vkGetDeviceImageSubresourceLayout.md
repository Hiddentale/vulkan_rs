# Usage Notes

Vulkan 1.4 command that queries the subresource layout for an image
**without creating it first**. Returns the offset, size, row pitch,
array pitch, and depth pitch for a given subresource of a
hypothetical image.

Useful for pre-planning host-side memory layouts when using
`HOST_TRANSFER` images, or for calculating buffer sizes for staging
uploads.
