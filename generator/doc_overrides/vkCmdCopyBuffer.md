# Usage Notes

Copies data between two buffers. Multiple regions can be copied in
a single call. Must be recorded outside a render pass.

Common patterns:

- **Staging upload**: copy from a host-visible staging buffer to a
  device-local buffer. This is the standard way to get vertex,
  index, and uniform data onto the GPU.
- **Buffer-to-buffer transfers**: defragment or reorganise GPU data.

Source and destination regions must not overlap within the same
buffer. Use a temporary staging buffer if you need to shift data
within a single buffer.

For Vulkan 1.3+, prefer `cmd_copy_buffer2` which uses an extensible
`CopyBufferInfo2` struct.
