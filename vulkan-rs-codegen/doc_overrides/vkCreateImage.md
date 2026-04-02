# Usage Notes

After creating an image you must bind memory to it before use:

1. Query requirements with `get_image_memory_requirements`.
2. Allocate from a compatible memory type with `allocate_memory`.
3. Bind with `bind_image_memory`.

Choose `IMAGE_TILING_OPTIMAL` for GPU-side textures and render targets.
Use `IMAGE_TILING_LINEAR` only when you need direct CPU access to the
texel layout (e.g. CPU readback), and check format support first with
`get_physical_device_image_format_properties`.

The `initial_layout` must be `UNDEFINED` or `PREINITIALIZED`.
Most applications use `UNDEFINED` and transition via a pipeline barrier.

Destroy with `destroy_image` when no longer needed. Do not destroy an
image that is still referenced by a framebuffer or image view.
