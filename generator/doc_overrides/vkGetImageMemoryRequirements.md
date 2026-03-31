# Usage Notes

Returns the memory requirements (size, alignment, compatible memory
type bits) for an image. Must be called before `bind_image_memory`.

Image memory requirements can differ significantly based on tiling,
format, and usage flags. An `IMAGE_TILING_OPTIMAL` image typically
requires `DEVICE_LOCAL` memory and has stricter alignment than a
linear image.

When sub-allocating linear and optimal images from the same memory
object, the `buffer_image_granularity` device limit applies. You may
need extra padding between the two to satisfy this constraint.

For Vulkan 1.1+, prefer `get_image_memory_requirements2` which
supports dedicated allocation queries.
