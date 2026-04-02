# Usage Notes

Vulkan 1.4 host-side image-to-image copy. Copies texel data between
two images from the CPU without recording a command buffer.

Both images must be in `GENERAL` layout and must have been created
with `HOST_TRANSFER` usage. The copy happens synchronously on the
calling thread.

Use cases are limited to CPU-side image manipulation (e.g. test
utilities, offline processing). For GPU-side copies in a render
loop, `cmd_copy_image2` is the standard path.
