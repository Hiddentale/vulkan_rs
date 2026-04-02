# Usage Notes

Destroys an image object. The image must not be in use by any
pending GPU work, and all image views referencing this image must
already be destroyed.

Destroying an image does **not** free its backing memory. Call
`free_memory` separately after destroying the image.

Safe teardown order for an image:

1. Wait for all GPU work using the image to complete.
2. Destroy all `ImageView` objects referencing the image.
3. Destroy any `Framebuffer` objects that included those views.
4. `destroy_image`.
5. Free or reclaim the backing memory.
