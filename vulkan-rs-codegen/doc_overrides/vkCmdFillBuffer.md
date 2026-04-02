# Usage Notes

Fills a region of a buffer with a repeating 4-byte value. Useful
for clearing GPU buffers to zero (or any uniform value) without a
staging upload.

Common uses:

- **Zero-initialise** an indirect draw count buffer before a compute
  culling pass.
- **Clear** a storage buffer used as a histogram or counter.

The `offset` and `size` must be multiples of 4. Use `VK_WHOLE_SIZE`
to fill from the offset to the end of the buffer.

Must be recorded outside a render pass. The buffer must have
`BUFFER_USAGE_TRANSFER_DST`.
