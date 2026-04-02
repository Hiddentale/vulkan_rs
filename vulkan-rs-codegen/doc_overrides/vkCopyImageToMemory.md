# Usage Notes

Vulkan 1.4 host-side image readback. Copies texel data from an
image directly to a host memory pointer without a staging buffer or
command buffer.

The image must be in `GENERAL` layout and must have been created
with `HOST_TRANSFER` usage. The copy happens synchronously.

This simplifies CPU readback workflows that previously required a
staging buffer + `cmd_copy_image_to_buffer` + fence wait + map.
However, it requires the image to support host transfer, which not
all implementations or formats support.
