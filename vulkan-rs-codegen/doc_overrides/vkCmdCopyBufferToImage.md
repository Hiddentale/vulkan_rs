# Usage Notes

Copies data from a buffer to an image, the primary way to upload
texture data from CPU to GPU.

**Typical upload workflow**:

1. Write pixel data into a host-visible staging buffer.
2. Transition the target image to `TRANSFER_DST_OPTIMAL`.
3. `cmd_copy_buffer_to_image` with the appropriate
   `BufferImageCopy` regions.
4. Transition the image to `SHADER_READ_ONLY_OPTIMAL` for sampling.

**Buffer layout**: `buffer_row_length` and `buffer_image_height`
control the row and slice pitch of the source data in the buffer.
Set both to zero to use a tightly packed layout matching the image
extent.

Multiple regions can be copied in a single call (e.g. all mip
levels of a texture). Must be recorded outside a render pass.

For Vulkan 1.3+, prefer `cmd_copy_buffer_to_image2`.
