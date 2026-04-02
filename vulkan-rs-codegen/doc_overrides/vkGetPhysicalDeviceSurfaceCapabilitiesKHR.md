# Usage Notes

Queries the surface capabilities for a physical device: supported
image count range, current extent, supported transforms, composite
alpha modes, and supported image usage flags.

**Key fields**:

- **`current_extent`**: the current surface size. If `0xFFFFFFFF`,
  the surface size is determined by the swapchain extent.
- **`min/max_image_count`**: the supported range for swapchain
  image count.
- **`current_transform`**: pass this as `pre_transform` in
  swapchain creation to avoid extra composition overhead.
- **`supported_usage_flags`**: which image usage bits the swapchain
  images support (always includes `COLOR_ATTACHMENT`).

Call this before creating a swapchain and again before recreating
after a resize, the capabilities may have changed.
