# Usage Notes

Vulkan 1.4 host-side image upload. Copies texel data from a host
memory pointer directly into an image without a staging buffer or
command buffer.

The image must be in `GENERAL` layout and must have been created
with `HOST_TRANSFER` usage. The copy happens synchronously.

This simplifies upload workflows that previously required a staging
buffer + `cmd_copy_buffer_to_image` + layout transitions. However,
the image must support host transfer and be in `GENERAL` layout,
which may not be optimal for subsequent GPU reads.
