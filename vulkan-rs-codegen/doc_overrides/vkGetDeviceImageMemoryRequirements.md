# Usage Notes

Vulkan 1.3 command that queries memory requirements for an image
**without creating it first**. Pass a `DeviceImageMemoryRequirements`
containing the hypothetical `ImageCreateInfo`.

Useful for pre-planning memory allocations or estimating VRAM usage
before committing to image creation.

For multi-planar images, set `plane_aspect` to query requirements
for a specific plane.

The returned requirements are identical to what
`get_image_memory_requirements2` would return for an actual image
created with the same parameters.
