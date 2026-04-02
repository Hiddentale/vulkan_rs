# Usage Notes

Copies data from an image to a buffer, used for GPU readback of
rendered images, screenshots, or compute shader output.

**Typical readback workflow**:

1. Transition the source image to `TRANSFER_SRC_OPTIMAL`.
2. `cmd_copy_image_to_buffer` into a host-visible buffer.
3. Submit and wait for the fence.
4. Map the buffer and read the pixel data on the CPU.

The `buffer_row_length` and `buffer_image_height` fields in
`BufferImageCopy` control the destination layout. Set both to zero
for tightly packed output.

Be aware that readback is not instantaneous, it requires a full
GPU round-trip. Avoid reading back in the render loop unless you
are double- or triple-buffering the readback to hide latency.

Must be recorded outside a render pass. For Vulkan 1.3+, prefer
`cmd_copy_image_to_buffer2`.
